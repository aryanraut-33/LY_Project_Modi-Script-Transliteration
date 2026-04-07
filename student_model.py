"""
==============================================================================
student_model.py — MoScNet Student Transformer Decoder (Novel Architecture)
==============================================================================

PURPOSE:
    Implements the paper's CORE CONTRIBUTION: the custom student transformer
    decoder with two architectural innovations:

    1. PARALLEL ATTENTION (from ViT-22B / PaLM)
       Instead of the standard sequential design:
           x → LayerNorm → Attention → Add → LayerNorm → FFN → Add → out
       MoScNet uses PARALLEL residual branches:
           x → LayerNorm → ┬── Attention ──┐
                            └── FFN ────────┘ → sum → Add → out
       Both self-attention and the feed-forward network operate on the SAME
       normalized input, and their outputs are summed. This improves training
       throughput and gradient flow while maintaining representational power.

    2. QK-NORMALIZATION (from ViT-22B)
       Before computing attention scores, the Query and Key vectors are
       normalized with LayerNorm:
           attn_scores = softmax( LayerNorm(Q) @ LayerNorm(K)^T / √d_k ) @ V
       This prevents attention logits from growing unboundedly at scale,
       stabilizing training especially for larger model variants.

    The paper (Section 4.1 — Student Model):
      "The student model architecture is inspired from [ViT-22B]. Unlike
       the full attention mechanism proposed in [ViT-22B], the student
       model incorporates causal attention to suit its task better."

    Ablation evidence (Paper Table 6):
      ┌────────────────────────────────────────────┬────────────┐
      │ Configuration                              │ BLEU score │
      ├────────────────────────────────────────────┼────────────┤
      │ MoScNet-XL (full: parallel attn + QK-norm) │   51.03    │
      │ Without parallel attention                 │   49.85    │
      │ Without QK-normalization                   │   50.21    │
      └────────────────────────────────────────────┴────────────┘

ARCHITECTURE (one decoder block):
    ┌─────────────────────────────────────────────────────────────────┐
    │  Input: x (B, L, d_model)                                     │
    │           ↓                                                    │
    │       RMSNorm(x)                                               │
    │       ╱         ╲                                              │
    │      ↓           ↓                                             │
    │  QK-Norm      SwiGLU                                           │
    │  Causal       Feed-Forward                                     │
    │  Attention    Network                                          │
    │      ↓           ↓                                             │
    │      └─── + ─────┘                                             │
    │           ↓                                                    │
    │       x + (attn_out + ffn_out)    ← Parallel residual          │
    │           ↓                                                    │
    │  Output: (B, L, d_model)                                      │
    └─────────────────────────────────────────────────────────────────┘

    Full model = N stacked decoder blocks + final RMSNorm + LM head

MODEL VARIANTS (Paper Table 4):
    ┌──────────┬────────┬────────┬───────┬────────┬────────────┐
    │ Variant  │ Hidden │ Blocks │ Heads │ d_ff   │ Params     │
    ├──────────┼────────┼────────┼───────┼────────┼────────────┤
    │ S        │   256  │    4   │   4   │  1024  │  ~15M      │
    │ M        │   512  │    8   │   8   │  2048  │  ~63M      │
    │ L        │   768  │   12   │  12   │  3072  │  ~177M     │
    │ XL       │  1024  │   16   │  16   │  4096  │  ~429M     │
    └──────────┴────────┴────────┴───────┴────────┴────────────┘

WHERE IT IS USED:
    - train.py (Phase 2) → receives Z_C, produces logits + hidden states
                           for computing the KD loss
    - evaluate.py         → autoregressive generation of Devanagari tokens

HOW IT INTERACTS WITH OTHERS:
    student_model.py is a pure PyTorch module with NO project dependencies
    (it only imports config for default hyperparameters).

    In the pipeline:
        language_encoder.fuse(Z_I, Z_L) → Z_C    (input)
        language_encoder.create_causal_mask()      (mask)
              ↓
        student_model(Z_C, mask) → (logits, hidden_states)
              ↓
        losses.py(student_logits, teacher_logits, ...) → loss
==============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import config


# ═══════════════════════════════════════════════════════════════════════════
# Building Blocks
# ═══════════════════════════════════════════════════════════════════════════


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Used instead of standard LayerNorm in modern LLMs (LLaMA, Gemma, etc.)
    because it's computationally cheaper (no mean subtraction, no bias)
    while providing similar training stability.

    Formula: x_norm = x / RMS(x) * weight
    where RMS(x) = sqrt(mean(x²) + eps)

    Args:
        d_model: Feature dimension to normalize over.
        eps:     Small constant for numerical stability.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)
        return x_normed * self.weight


class QKNormCausalAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention with QK-Normalization.

    This is one of the paper's two architectural novelties. Before computing
    attention scores, the Q and K vectors are normalized with LayerNorm per
    head. This stabilizes attention logits and prevents them from growing
    unboundedly, which is especially important at larger scales.

    Standard attention:   attn = softmax(Q @ K^T / √d_k) @ V
    QK-Norm attention:    attn = softmax(LN(Q) @ LN(K)^T / √d_k) @ V

    The paper (Section 6.4 — Ablation):
      "Omitting QK-Norm decreases the BLEU score, suggesting that
       normalizing the attention keys and queries helps stabilize the
       attention mechanism, improving feature alignment."

    Args:
        d_model:   Total embedding dimension.
        num_heads: Number of attention heads (d_model must be divisible).
        dropout:   Attention dropout rate.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Projection matrices
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # QK-Normalization: per-head LayerNorm on Q and K
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x:    (B, L, d_model) — input embeddings.
            mask: (1, 1, L, L)   — attention mask (1=attend, 0=mask).

        Returns:
            (B, L, d_model) — attention output.
        """
        B, L, D = x.shape

        # Project to Q, K, V and reshape for multi-head
        # (B, L, d_model) → (B, num_heads, L, head_dim)
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # ── QK-Normalization ─────────────────────────────────────────
        # Normalize Q and K per head before computing attention scores.
        # This is what makes attention stable at scale.
        q = self.q_norm(q)
        k = self.k_norm(k)

        # ── Scaled Dot-Product Attention ─────────────────────────────
        # (B, H, L, head_dim) @ (B, H, head_dim, L) → (B, H, L, L)
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        # Apply causal mask (mask out future tokens)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # (B, H, L, L) @ (B, H, L, head_dim) → (B, H, L, head_dim)
        context = torch.matmul(attn_weights, v)

        # Merge heads: (B, H, L, head_dim) → (B, L, d_model)
        context = context.transpose(1, 2).contiguous().view(B, L, D)

        # Output projection + residual dropout
        output = self.o_proj(context)
        return self.resid_dropout(output)


class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network.

    A gated linear unit with SiLU (Swish) activation, used in modern LLMs
    (LLaMA, PaLM, Gemma) instead of the standard ReLU or GELU FFN.

    Formula: SwiGLU(x) = W3 · (SiLU(W1 · x) ⊙ W2 · x)

    The gating mechanism (W2 branch) provides the model with more expressive
    power per parameter compared to a simple Linear→ReLU→Linear FFN.

    Args:
        d_model: Input/output dimension.
        d_ff:    Hidden (expanded) dimension (typically 4× d_model).
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)  # gate projection
        self.w2 = nn.Linear(d_model, d_ff, bias=False)  # up projection
        self.w3 = nn.Linear(d_ff, d_model, bias=False)  # down projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


# ═══════════════════════════════════════════════════════════════════════════
# Decoder Block (with Parallel Attention)
# ═══════════════════════════════════════════════════════════════════════════


class MoScNetDecoderBlock(nn.Module):
    """
    Single decoder block with PARALLEL attention and feed-forward.

    This is the paper's first architectural novelty. Instead of running
    self-attention and FFN sequentially:
        out = x + FFN(LayerNorm(x + Attention(LayerNorm(x))))  [sequential]

    We run them in PARALLEL from the same normalized input:
        out = x + Attention(RMSNorm(x)) + FFN(RMSNorm(x))     [parallel]

    Why parallel? (from PaLM / ViT-22B):
      - Better gradient flow (both branches contribute independently)
      - ~15% faster training (parallelizable computation)
      - No loss in representational power (residual preserves input)

    The paper (Section 6.4 — Ablation):
      "The removal of parallel attention reduces the BLEU score, indicating
       that parallel attention layers more effectively capture and integrate
       long-range dependencies."

    Args:
        d_model:   Feature dimension.
        num_heads: Number of attention heads.
        d_ff:      FFN hidden dimension.
        dropout:   Dropout rate.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.attn = QKNormCausalAttention(d_model, num_heads, dropout)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Parallel residual forward pass.

        Args:
            x:    (B, L, d_model) — input from previous block.
            mask: (1, 1, L, L)   — causal attention mask.

        Returns:
            (B, L, d_model) — output to next block.
        """
        # Single normalization shared by both branches
        normed = self.norm(x)

        # Parallel computation: attention and FFN operate on the SAME input
        attn_out = self.attn(normed, mask)
        ffn_out = self.ffn(normed)

        # Parallel residual connection
        return x + attn_out + ffn_out


# ═══════════════════════════════════════════════════════════════════════════
# Full Student Model
# ═══════════════════════════════════════════════════════════════════════════


class MoScNetStudent(nn.Module):
    """
    The complete MoScNet Student Transformer.

    Architecture:
        Z_C → [DecoderBlock × N] → RMSNorm → Linear (LM head) → logits

    The input Z_C is the fused vision-language embedding from
    language_encoder.fuse(Z_I, Z_L). The model processes this sequence
    autoregressively to predict Devanagari tokens.

    During training (Phase 2), the model also outputs hidden states
    for computing the L2 distillation loss against the teacher.

    Args:
        hidden_dim: Model hidden dimension (d_model of student, NOT teacher).
        num_blocks: Number of stacked decoder blocks.
        num_heads:  Number of attention heads per block.
        d_ff:       FFN hidden dimension.
        vocab_size: Output vocabulary size (must match teacher tokenizer).
        dropout:    Dropout rate.
    """

    def __init__(
        self,
        hidden_dim: int = config.student_hidden_dim,
        num_blocks: int = config.student_num_blocks,
        num_heads: int = config.student_num_heads,
        d_ff: int = config.student_d_ff,
        vocab_size: int = config.teacher_vocab_size,
        dropout: float = config.student_dropout,
    ):
        super().__init__()

        # ── Input projection ─────────────────────────────────────────
        # Z_C comes from language_encoder at d_model (teacher dim, e.g. 4096)
        # Student operates at student_hidden_dim (e.g. 512 for variant M)
        # This projection bridges the two spaces
        self.input_proj = nn.Linear(config.d_model, hidden_dim, bias=False)

        # ── Stacked decoder blocks ───────────────────────────────────
        self.blocks = nn.ModuleList([
            MoScNetDecoderBlock(hidden_dim, num_heads, d_ff, dropout)
            for _ in range(num_blocks)
        ])

        # ── Output layers ────────────────────────────────────────────
        self.final_norm = RMSNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

        # ── Hidden state projection (for L2 loss) ────────────────────
        # Maps student hidden dim back to teacher dim for feature distillation
        self.hidden_proj = nn.Linear(hidden_dim, config.d_model, bias=False)

        # Store config for reference
        self._config = {
            "hidden_dim": hidden_dim,
            "num_blocks": num_blocks,
            "num_heads": num_heads,
            "d_ff": d_ff,
            "vocab_size": vocab_size,
        }

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> tuple:
        """
        Forward pass through the student model.

        Args:
            x:    (B, L, d_model) — fused embeddings Z_C from language_encoder.
                  NOTE: d_model here is the TEACHER's dimension (e.g. 4096).
            mask: (1, 1, L, L)   — causal attention mask.

        Returns:
            logits:        (B, L, vocab_size) — next-token prediction scores.
            hidden_state:  (B, L, d_model)    — projected to teacher dim for
                           L2 loss computation in losses.py.
        """
        # Project from teacher dim to student dim
        x = self.input_proj(x)  # (B, L, student_hidden_dim)

        # Pass through stacked decoder blocks
        for block in self.blocks:
            x = block(x, mask)

        # Final normalization
        x_normed = self.final_norm(x)

        # LM head for next-token prediction
        logits = self.lm_head(x_normed)  # (B, L, vocab_size)

        # Project hidden states to teacher dim for L2 distillation loss
        hidden_state = self.hidden_proj(x_normed)  # (B, L, teacher_d_model)

        return logits, hidden_state
