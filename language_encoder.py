"""
==============================================================================
language_encoder.py — Tokenization, Embedding, and Multi-Modal Fusion
==============================================================================

PURPOSE:
    This module handles the text side of the VLM pipeline and the critical
    fusion step where vision and language meet:

    1. TOKENIZATION: Converts Devanagari text into BPE token IDs using the
       teacher LLM's tokenizer (no custom tokenizer needed).

    2. EMBEDDING: Maps token IDs into dense vectors in the LLM's embedding
       space, producing language embeddings Z_L.

    3. FUSION: Concatenates image embeddings Z_I (from vision_encoder.py)
       with language embeddings Z_L to form the combined input Z_C that
       feeds into both the teacher and student models.

    4. MASKING: Builds the causal attention mask that enforces:
       - Vision tokens can attend to ALL other vision tokens (bidirectional)
       - Language tokens can attend to ALL vision tokens AND previous
         language tokens only (causal / autoregressive)

    The paper (Section 4.1 — Language Encoder):
      "The Devanagari text transliteration T is first processed by a
       pre-trained tokenizer, which splits the sequence into tokens.
       We employed the same BPE tokenizer used by our teacher model
       without introducing any additional preprocessing steps for
       the Devanagari script."

    Why no custom tokenizer? (Section 4.1):
      "LLMs such as LLaMA-2, LLaMA-3, and Phi-3 Mini already incorporate
       training on Hindi data — an Indian language that shares the same
       Devanagari script as Marathi. Because Marathi and Hindi share the
       same orthographic system, the existing vocabulary and tokenization
       scheme capture Marathi text significantly."

ARCHITECTURE:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                                                                     │
    │  Devanagari Text ──→ BPE Tokenizer ──→ token IDs                  │
    │                                            ↓                       │
    │                                     Embedding Layer                │
    │                                            ↓                       │
    │                                    Z_L (B, T, d_model)             │
    │                                            ↓                       │
    │  Z_I (B, P, d_model) ──────→ concat ──→ Z_C (B, P+T, d_model)    │
    │  (from vision_encoder.py)                  ↓                       │
    │                                     Causal Mask                    │
    │                                            ↓                       │
    │                              teacher_model / student_model         │
    └─────────────────────────────────────────────────────────────────────┘
    where P = num_patches (196 for SigLIP-Base), T = text token count

WHERE IT IS USED:
    - train.py    → tokenize + embed text, fuse with vision, build mask
    - evaluate.py → tokenize prompts, fuse with vision for generation

HOW IT INTERACTS WITH OTHERS:
    - Receives Z_I from vision_encoder.py
    - Produces Z_C + mask consumed by teacher_model.py and student_model.py
    - Tokenizer is shared with data_loader.py (same tokenizer instance)
    - Embedding weights may be initialized from or shared with teacher_model.py

    Flow:
        vision_encoder.py → Z_I ─┐
                                  ├──→ language_encoder.fuse() → Z_C
        language_encoder.py → Z_L ┘                                ↓
                                                        teacher / student
==============================================================================
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from config import config


class MoScNetLanguageEncoder(nn.Module):
    """
    Tokenizer + embedding layer + multi-modal fusion for MoScNet.

    This module wraps the teacher LLM's BPE tokenizer and provides:
      - tokenize():         text → token IDs
      - forward():          token IDs → embeddings Z_L
      - fuse():             (Z_I, Z_L) → Z_C
      - create_causal_mask(): builds the vision-language attention mask

    Args:
        tokenizer_name: HuggingFace model ID for the tokenizer.
                        Must match the teacher model for vocabulary consistency.
    """

    def __init__(self, tokenizer_name: str = config.teacher_model_name):
        super().__init__()

        # ── Initialize tokenizer ─────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # LLaMA-3 doesn't have a pad token by default — use EOS
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # ── Embedding layer ──────────────────────────────────────────
        # This maps token IDs to d_model vectors. In practice, we'll
        # initialize these weights from the teacher model's embedding
        # layer for better starting representations.
        self.embedding = nn.Embedding(
            num_embeddings=len(self.tokenizer),
            embedding_dim=config.d_model,
        )

    def tokenize(self, text: str) -> dict:
        """
        Tokenize a Devanagari text string into token IDs.

        Args:
            text: Cleaned Devanagari string (from preprocessing.py).

        Returns:
            Dict with "input_ids" and "attention_mask" tensors.
        """
        return self.tokenizer(
            text,
            max_length=config.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Embed token IDs into the LLM's vector space.

        Args:
            input_ids: (B, T) integer token IDs.

        Returns:
            Z_L: (B, T, d_model) language embeddings.
        """
        return self.embedding(input_ids)

    def fuse(self, Z_I: torch.Tensor, Z_L: torch.Tensor) -> torch.Tensor:
        """
        Concatenate vision and language embeddings along the sequence axis.

        The image tokens are placed FIRST (as a prefix), followed by the
        text tokens. This means the language model "sees" the image before
        generating text — exactly how LLaVA and MoScNet work.

        Args:
            Z_I: (B, P, d_model) — image patch embeddings from vision_encoder.
            Z_L: (B, T, d_model) — text token embeddings.

        Returns:
            Z_C: (B, P + T, d_model) — combined multi-modal embeddings.
        """
        return torch.cat([Z_I, Z_L], dim=1)

    def create_causal_mask(
        self,
        num_vision: int,
        num_lang: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Build the attention mask for the fused vision-language sequence.

        Mask structure (total_len = num_vision + num_lang):

            ┌─────────────────────┬──────────────────────┐
            │  Vision-to-Vision   │  Vision-to-Language  │
            │   (ALL ONES)        │   (ALL ONES)         │  ← Vision rows
            ├─────────────────────┼──────────────────────┤
            │  Lang-to-Vision     │  Lang-to-Language    │
            │   (ALL ONES)        │   (CAUSAL / TRIL)    │  ← Language rows
            └─────────────────────┴──────────────────────┘

        - Vision tokens see everything (including future language tokens,
          though during training with teacher forcing this doesn't cause
          information leakage since vision tokens don't predict text).
        - Language tokens see ALL vision tokens (full cross-attention to
          the image) but only PAST language tokens (causal self-attention).

        Args:
            num_vision: Number of image patch tokens (P).
            num_lang:   Number of text tokens (T).
            device:     torch.device for tensor placement.

        Returns:
            mask: (1, 1, P+T, P+T) — broadcastable attention mask.
                  1 = attend, 0 = mask out.
        """
        total_len = num_vision + num_lang

        # Start with all ones (everything visible)
        mask = torch.ones((total_len, total_len), device=device)

        # Apply causal mask to language-to-language attention block
        causal_block = torch.tril(
            torch.ones((num_lang, num_lang), device=device)
        )
        mask[num_vision:, num_vision:] = causal_block

        # Expand for batch dim and num_heads dim: (1, 1, total, total)
        return mask.unsqueeze(0).unsqueeze(0)

    def init_from_teacher(self, teacher_embedding_weight: torch.Tensor) -> None:
        """
        Initialize the embedding layer with the teacher model's weights.

        This gives the student model a strong starting point for text
        representation, rather than learning embeddings from scratch.

        Args:
            teacher_embedding_weight: (vocab_size, d_model) tensor from
                                      the teacher LLM's embedding layer.
        """
        with torch.no_grad():
            self.embedding.weight.copy_(teacher_embedding_weight)
