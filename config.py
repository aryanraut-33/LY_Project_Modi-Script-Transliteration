"""
==============================================================================
config.py — Central Configuration for MoScNet VLM Re-Implementation
==============================================================================

PURPOSE:
    This file is the single source of truth for ALL hyperparameters, paths,
    model choices, and hardware settings used across the entire pipeline.
    Nothing is hard-coded in any other file; every module imports from here.

WHERE IT IS USED:
    - data_loader.py    → dataset name, batch size, max sequence length, split seed
    - preprocessing.py  → image size, normalization stats
    - vision_encoder.py → encoder model name, hidden dimension
    - language_encoder.py → teacher model name (for tokenizer), d_model
    - teacher_model.py  → model name, LoRA rank/alpha, quantization flags
    - student_model.py  → hidden dim, num blocks, num heads, FFN dim, dropout
    - losses.py         → loss weights (alpha, beta, gamma), KL temperature
    - train.py          → learning rates, epochs, gradient accumulation, precision
    - evaluate.py       → checkpoint path, decoding settings
    - utils.py          → log/checkpoint directories, seed

HOW IT INTERACTS WITH OTHERS:
    Every module does `from config import config` and reads the fields it needs.
    To change any setting (e.g., switch student variant from M to XL, or change
    the teacher LLM), you modify ONLY this file — nothing else needs to change.

HARDWARE TARGET:
    Google Colab Free Tier — NVIDIA T4 (15 GB VRAM, FP16, no BF16 support).
    All defaults are tuned to fit within this constraint.

PAPER REFERENCE:
    Kausadikar et al., "Historic Scripts to Modern Vision: A Novel Dataset
    and A VLM Framework for Transliteration of Modi Script to Devanagari"
    (ICDAR 2025). Values are drawn from Section 4 and Section 5.
==============================================================================
"""

import torch
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Config:
    """
    All hyperparameters for the MoScNet re-implementation.
    Grouped by pipeline stage for easy navigation.
    """

    # ── Experiment Metadata ──────────────────────────────────────────────
    experiment_name: str = "moscnet_vlm_reimpl"
    seed: int = 42

    # ── Hardware & Precision (Colab T4) ──────────────────────────────────
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16: bool = True          # T4 supports FP16 but NOT BF16
    use_4bit: bool = True          # 4-bit quantization for teacher LLM
    use_gradient_checkpointing: bool = True  # saves ~30% VRAM

    # ── Dataset ──────────────────────────────────────────────────────────
    dataset_name: str = "historyHulk/MoDeTrans"
    synth_dataset_name: str = "historyHulk/SynthMoDe"  # optional
    max_seq_len: int = 512         # max Devanagari token sequence length
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1

    # ── Image Settings ───────────────────────────────────────────────────
    # Paper uses 128×256, but SigLIP expects square input (224×224)
    img_size: Tuple[int, int] = (224, 224)
    img_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5)  # SigLIP norm
    img_std: Tuple[float, float, float] = (0.5, 0.5, 0.5)

    # ── Vision Encoder (SigLIP-Base) ─────────────────────────────────────
    # Paper's best: SigLIP-Large. We use Base to fit on T4.
    vision_encoder_name: str = "google/siglip-base-patch16-224"
    vision_hidden_dim: int = 768   # output dim of siglip-base

    # ── Teacher Model (LLaMA-3 8B, 4-bit) ───────────────────────────────
    # Paper's best: LLaMA-3 70B. We use 8B quantized to fit on T4.
    teacher_model_name: str = "unsloth/llama-3-8b-bnb-4bit"
    d_model: int = 4096            # LLaMA-3 8B hidden dimension
    teacher_vocab_size: int = 128256  # LLaMA-3 tokenizer vocab size

    # ── LoRA Settings ────────────────────────────────────────────────────
    lora_r: int = 64               # rank (paper uses 64)
    lora_alpha: int = 128          # alpha = 2× rank (standard)
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # ── Student Model Variants ───────────────────────────────────────────
    # Paper Table 4: S/M/L/XL configurations
    student_variant: str = "M"     # start with Medium for T4

    # Variant-specific settings (set automatically in __post_init__)
    student_hidden_dim: int = 512
    student_num_blocks: int = 8
    student_num_heads: int = 8
    student_d_ff: int = 2048       # typically 4× hidden_dim
    student_dropout: float = 0.1

    # ── Training Hyperparameters ─────────────────────────────────────────
    lr_teacher: float = 1e-6       # paper: 1e-6 for teacher
    lr_student: float = 1e-4       # paper: 1e-4 for student
    weight_decay: float = 0.01     # AdamW
    warmup_steps: int = 100
    epochs_teacher: int = 10       # Phase 1: LoRA fine-tuning
    epochs_student: int = 20       # Phase 2: Knowledge distillation
    batch_size: int = 1            # T4 constraint — single sample
    grad_accum_steps: int = 32     # effective batch = 32
    num_workers: int = 2           # Colab has limited CPUs
    max_grad_norm: float = 1.0     # gradient clipping

    # ── Distillation Loss Weights ────────────────────────────────────────
    # Paper Table 5: all three losses together → BLEU 51.03
    alpha_ce: float = 1.0          # cross-entropy (primary task)
    beta_l2: float = 1.0           # L2 norm (feature-level KD)
    gamma_kl: float = 1.0          # KL-divergence (distribution-level KD)
    kl_temperature: float = 2.0    # softens teacher distributions

    # ── Paths ────────────────────────────────────────────────────────────
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    def __post_init__(self):
        """Auto-configure student dimensions based on variant choice."""
        variants = {
            "S":  {"hidden": 256,  "blocks": 4,  "heads": 4,  "d_ff": 1024},
            "M":  {"hidden": 512,  "blocks": 8,  "heads": 8,  "d_ff": 2048},
            "L":  {"hidden": 768,  "blocks": 12, "heads": 12, "d_ff": 3072},
            "XL": {"hidden": 1024, "blocks": 16, "heads": 16, "d_ff": 4096},
        }
        if self.student_variant in variants:
            v = variants[self.student_variant]
            self.student_hidden_dim = v["hidden"]
            self.student_num_blocks = v["blocks"]
            self.student_num_heads  = v["heads"]
            self.student_d_ff      = v["d_ff"]


# ── Global singleton ─────────────────────────────────────────────────────
# Import this from any module: `from config import config`
config = Config()
