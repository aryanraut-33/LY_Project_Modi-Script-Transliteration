"""
==============================================================================
vision_encoder.py — SigLIP Vision Encoder + MLP Projector for MoScNet
==============================================================================

PURPOSE:
    Converts Modi script images into language-space embeddings that can be
    concatenated with text embeddings and fed into the teacher/student LLMs.

    This module implements the first half of the Vision-Language bridge:
        Image → SigLIP (frozen) → patch tokens → MLP projector (trainable) → Z_I

    The paper (Section 4.1 — Vision Encoder):
      "A vision encoder processes and converts input Modi script image M
       into image representations I. These image representations are then
       passed to a trainable MLP layer. The MLP layer transforms I into
       language embeddings Z_I."

    SigLIP was chosen because (Section 6.1 — Ablation):
      "SigLIP consistently outperformed other vision encoders [CLIP, ViT-L,
       SwinTransformer]. SigLIP's ability to represent image features as
       discrete, semantically meaningful tokens... facilitates more robust
       cross-modal alignment."

ARCHITECTURE:
    ┌──────────────────────────────────────────────────────────────────┐
    │  Input: pixel_values (B, 3, 224, 224)                          │
    │                    ↓                                            │
    │  SigLIP ViT (FROZEN)                                           │
    │  - Patch embedding: 224/16 = 14×14 = 196 patches               │
    │  - Each patch → 768-dim vector (siglip-base)                   │
    │  - Output: (B, 196, 768)                                       │
    │                    ↓                                            │
    │  MLP Projector (TRAINABLE)                                     │
    │  - Linear(768, 4096) → GELU → Linear(4096, 4096)              │
    │  - Projects vision features into LLM's embedding space         │
    │  - Output: Z_I = (B, 196, 4096)                               │
    └──────────────────────────────────────────────────────────────────┘

WHERE IT IS USED:
    - train.py    → called on every batch to encode images before fusion
    - evaluate.py → called during inference to encode test images

HOW IT INTERACTS WITH OTHERS:
    vision_encoder.py sits between preprocessing.py and language_encoder.py:

        preprocessing.py
            ↓  pixel_values (B, 3, 224, 224)
        vision_encoder.py
            ↓  Z_I (B, num_patches, d_model)
        language_encoder.py  ──→  Z_C = concat(Z_I, Z_L)
            ↓
        teacher_model.py / student_model.py

    The MLP projector's output dimension (d_model) MUST match the teacher
    LLM's hidden dimension (4096 for LLaMA-3 8B) so that image and text
    embeddings live in the same vector space.
==============================================================================
"""

import torch
import torch.nn as nn
from transformers import SiglipVisionModel
from config import config


class MLPProjector(nn.Module):
    """
    Two-layer MLP that projects vision encoder outputs into the LLM's
    embedding space.

    Architecture: Linear → GELU → Linear
    This is identical to the projector used in LLaVA (Liu et al., 2023),
    which MoScNet's architecture is inspired by.

    Args:
        input_dim:  Dimension of vision encoder output (768 for SigLIP-Base).
        output_dim: Dimension of LLM hidden space (4096 for LLaMA-3 8B).
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_patches, input_dim)
        Returns:
            (B, num_patches, output_dim)
        """
        return self.net(x)


class MoScNetVisionEncoder(nn.Module):
    """
    Frozen SigLIP vision encoder + trainable MLP projector.

    The vision encoder extracts patch-level features from Modi script images.
    All SigLIP weights are frozen — only the MLP projector is trained.
    This follows the paper's approach where the vision encoder is not
    fine-tuned, keeping training efficient.

    Args:
        model_name: HuggingFace model ID for SigLIP. Default from config.
        output_dim: Target embedding dimension (must match LLM's d_model).
    """

    def __init__(
        self,
        model_name: str = config.vision_encoder_name,
        output_dim: int = config.d_model,
    ):
        super().__init__()

        # ── Load pretrained SigLIP ───────────────────────────────────
        self.vision_model = SiglipVisionModel.from_pretrained(model_name)

        # ── Freeze all vision encoder parameters ─────────────────────
        # Paper: vision encoder weights are NOT updated during training
        for param in self.vision_model.parameters():
            param.requires_grad = False

        # ── Trainable MLP projector ──────────────────────────────────
        # Maps from SigLIP hidden dim → LLM hidden dim
        self.projector = MLPProjector(config.vision_hidden_dim, output_dim)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encode images into language-space embeddings.

        Args:
            pixel_values: (B, 3, H, W) — preprocessed and normalized images.

        Returns:
            Z_I: (B, num_patches, d_model) — image embeddings in LLM space.
                 For SigLIP-Base with patch_size=16 and input 224×224:
                 num_patches = (224/16)² = 196
        """
        # SigLIP forward pass (frozen)
        # Returns: last_hidden_state of shape (B, num_patches, vision_hidden_dim)
        with torch.no_grad():
            outputs = self.vision_model(pixel_values=pixel_values)
        last_hidden_state = outputs.last_hidden_state

        # Project to LLM embedding space (trainable)
        Z_I = self.projector(last_hidden_state)
        return Z_I

    def train(self, mode: bool = True):
        """
        Override train() to keep the vision model frozen in eval mode.
        Only the MLP projector switches between train/eval.
        """
        super().train(mode)
        self.vision_model.eval()  # always eval — no dropout/BN changes
        return self
