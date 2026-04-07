"""
==============================================================================
teacher_model.py — LoRA-Adapted Teacher LLM for MoScNet
==============================================================================

PURPOSE:
    Wraps a pretrained Large Language Model (LLM) with Low-Rank Adaptation
    (LoRA) to serve as the teacher in MoScNet's Knowledge Distillation
    framework.

    The teacher has two roles across two training phases:

    PHASE 1 — LoRA Fine-Tuning:
      The teacher learns the Modi→Devanagari transliteration task by
      fine-tuning only the LoRA adapter weights (rank-64 matrices) while
      keeping the base LLM frozen. This produces high-quality soft targets
      with minimal compute.

    PHASE 2 — Knowledge Distillation:
      The teacher is completely frozen. It provides:
        - Soft logits  → for KL-divergence loss (distribution-level KD)
        - Hidden states (E_1) → for L2 norm loss (feature-level KD)
      The student learns to mimic these outputs.

    The paper (Section 4.1 — Teacher Model):
      "The teacher model is a pretrained LLM, fine-tuned for the task of
       transliterating Modi Script images into Devanagari text. Instead of
       directly fine-tuning by updating its weights, we adopt LoRA. The
       LoRA technique freezes weight matrices and introduces trainable
       low-rank decomposition matrices at each layer."

    LoRA Mechanics:
      For each target weight matrix W ∈ R^(d×d):
        W_adapted = W_frozen + B @ A
      where B ∈ R^(d×r), A ∈ R^(r×d), and r << d (rank = 64).
      Only B and A are trained — this adds <1% trainable parameters.

HARDWARE ADAPTATION:
    Paper uses LLaMA-3 70B on 128×H100 GPUs.
    We use LLaMA-3 8B with 4-bit NF4 quantization on a single T4 (15GB).
      - 4-bit quantization reduces 8B params from ~16GB to ~5.5GB VRAM
      - Double quantization (quantize the quantization constants) saves more
      - NF4 data type is optimized for normally-distributed neural net weights

WHERE IT IS USED:
    - train.py (Phase 1) → LoRA fine-tuning with teacher_ce_loss
    - train.py (Phase 2) → frozen forward pass for KD targets

HOW IT INTERACTS WITH OTHERS:
    teacher_model.py receives Z_C (fused embeddings) from language_encoder.py
    and produces logits + hidden states consumed by losses.py:

        language_encoder.fuse(Z_I, Z_L) → Z_C    (input)
              ↓
        teacher_model(Z_C) → (logits, E_1)
              ↓                    ↓
        losses.teacher_ce_loss()  losses.moscnet_kd_loss()
              (Phase 1)                (Phase 2)

    Dependencies:
      - transformers (AutoModelForCausalLM)
      - peft (LoraConfig, get_peft_model)
      - bitsandbytes (4-bit quantization)
      - config.py (model name, LoRA settings, quantization flags)
==============================================================================
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from config import config


class MoScNetTeacher(nn.Module):
    """
    Teacher LLM with 4-bit quantization and LoRA adapters.

    Lifecycle:
      1. __init__(): Load quantized LLM, inject LoRA, enable grad checkpointing
      2. Phase 1:    Train LoRA weights with teacher_ce_loss (train.py)
      3. freeze():   Lock all parameters for Phase 2
      4. Phase 2:    Forward-only, producing soft targets for student

    Args:
        model_name: HuggingFace model ID. Default: unsloth/llama-3-8b-bnb-4bit.
    """

    def __init__(self, model_name: str = config.teacher_model_name):
        super().__init__()

        # ── 4-bit Quantization Config ────────────────────────────────
        # NF4 (Normal Float 4-bit) is optimized for neural net weights
        # Double quantization further compresses the quantization constants
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=config.use_4bit,
            bnb_4bit_compute_dtype=torch.float16,  # T4 supports FP16
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        # ── Load Base Model ──────────────────────────────────────────
        print(f"Loading teacher model: {model_name}")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",       # auto-place layers across GPU/CPU
            trust_remote_code=True,
        )

        # ── Prepare for QLoRA Training ───────────────────────────────
        # This handles quantized model quirks (disabling cache, casting norms)
        self.base_model = prepare_model_for_kbit_training(self.base_model)

        # ── LoRA Configuration ───────────────────────────────────────
        # Paper: rank=64, applied to q_proj and v_proj
        # We extend to all projection matrices for better adaptation
        lora_cfg = LoraConfig(
            r=config.lora_r,                        # rank = 64
            lora_alpha=config.lora_alpha,            # alpha = 128 (2× rank)
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # ── Wrap with LoRA ───────────────────────────────────────────
        self.model = get_peft_model(self.base_model, lora_cfg)

        # ── Gradient Checkpointing ───────────────────────────────────
        # Trades ~30% speed for ~30% VRAM savings (critical for T4)
        if config.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Print trainable parameter count
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Teacher LoRA params: {trainable:,} / {total:,} "
              f"({100 * trainable / total:.2f}% trainable)")

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> tuple:
        """
        Forward pass through the LoRA-adapted teacher.

        IMPORTANT: This accepts embeddings (not token IDs) because the
        input is the fused Z_C from language_encoder, which already
        combines image patch embeddings and text token embeddings.

        Args:
            inputs_embeds:  (B, L, d_model) — fused Z_C embeddings.
            attention_mask: (B, L)          — 1=real token, 0=padding.

        Returns:
            logits:       (B, L, vocab_size) — next-token prediction scores.
            hidden_state: (B, L, d_model)    — last layer hidden states (E_1)
                          for L2 distillation loss.
        """
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        logits = outputs.logits                # (B, L, vocab_size)
        hidden_state = outputs.hidden_states[-1]  # (B, L, d_model) — last layer

        return logits, hidden_state

    def freeze(self) -> None:
        """
        Completely freeze the teacher for Phase 2 (Knowledge Distillation).

        After Phase 1 LoRA training is complete, call this to:
          1. Set all parameters to requires_grad=False
          2. Switch to eval mode (disables dropout, etc.)
          3. Disable gradient checkpointing (not needed for inference)

        The teacher then serves as a frozen oracle producing soft targets.
        """
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        # Disable gradient checkpointing for faster inference
        if hasattr(self.model, 'gradient_checkpointing_disable'):
            self.model.gradient_checkpointing_disable()
        print("Teacher model frozen for Knowledge Distillation.")

    def get_embedding_weight(self) -> torch.Tensor:
        """
        Extract the teacher's token embedding weights.

        Used by language_encoder.init_from_teacher() to initialize
        the student's embedding layer with the teacher's learned
        representations.

        Returns:
            (vocab_size, d_model) embedding weight tensor.
        """
        return self.model.get_input_embeddings().weight.data.clone()
