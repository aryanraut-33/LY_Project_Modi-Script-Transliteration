"""
==============================================================================
losses.py — Knowledge Distillation Loss Functions for MoScNet
==============================================================================

PURPOSE:
    Implements the three-component loss function used to train the MoScNet
    student model via Knowledge Distillation from the teacher LLM.

    The paper (Section 4.1 — Student Model) defines the overall loss as:

        L = α · L_CE^S  +  β · L_L2  +  γ · L_DKL

    where:
      1. L_CE^S  — Student Cross-Entropy Loss (primary task loss)
         Standard next-token prediction loss between student predictions
         and ground-truth Devanagari tokens. This teaches the student
         the transliteration task directly.

      2. L_L2    — L2 Norm Loss (feature-level distillation)
         Euclidean distance between teacher hidden states E_1 and student
         hidden states E_2. This forces the student's internal representations
         to align with the teacher's, transferring the teacher's learned
         feature space.

      3. L_DKL   — KL-Divergence Loss (distribution-level distillation)
         KL-divergence between softened teacher and student output
         distributions. Using temperature scaling (T > 1) reveals the
         teacher's knowledge about inter-class relationships (e.g., which
         Devanagari characters the teacher considers "close" to the
         correct one).

    Ablation evidence (Paper Table 5):
      ┌─────────────────────────────────────┬────────────┐
      │ Loss combination                    │ BLEU score │
      ├─────────────────────────────────────┼────────────┤
      │ L_CE only                           │   48.02    │
      │ L_CE + L_L2                         │   49.71    │
      │ L_CE + L_DKL                        │   50.12    │
      │ L_CE + L_L2 + L_DKL (full MoScNet) │   51.03    │
      └─────────────────────────────────────┴────────────┘

WHERE IT IS USED:
    - train.py (Phase 1) → teacher_ce_loss() to train teacher's LoRA weights
    - train.py (Phase 2) → moscnet_kd_loss() to train the student model

HOW IT INTERACTS WITH OTHERS:
    losses.py is a pure computation module. It receives tensors from the
    teacher and student models and returns scalar loss values.

    Data flow:
        teacher_model.py → teacher_logits, E_1 (hidden states)
        student_model.py → student_logits, E_2 (hidden states)
        data_loader.py   → labels (ground truth token IDs)
              ↓
        losses.py → scalar loss → .backward() in train.py

    No project imports — this module depends only on PyTorch.
==============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config


def teacher_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Cross-entropy loss for Phase 1 teacher LoRA fine-tuning.

    This is the standard causal language modeling loss. The teacher
    learns to predict the next Devanagari token given the Modi script
    image + preceding tokens.

    The loss function handles the shift internally:
        prediction[t] is compared to label[t+1]

    Args:
        logits: (B, T, vocab_size) — teacher's output logits.
        labels: (B, T) — ground truth token IDs. Pad positions = -100.

    Returns:
        Scalar loss tensor.
    """
    # Shift: predict token t+1 from position t
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    return loss_fn(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )


def student_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Cross-entropy loss for the student model (L_CE^S in the paper).

    Identical to teacher_ce_loss but kept separate for clarity
    and potential future modifications (e.g., label smoothing).

    Args:
        logits: (B, T, vocab_size) — student's output logits.
        labels: (B, T) — ground truth token IDs. Pad positions = -100.

    Returns:
        Scalar loss tensor.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    return loss_fn(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )


def l2_loss(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
) -> torch.Tensor:
    """
    L2 Norm Loss for feature-level Knowledge Distillation (L_L2 in paper).

    Measures the Euclidean distance between teacher and student hidden
    representations. This forces the student's internal feature space
    to mimic the teacher's.

    The paper states:
      "L2 norm loss measures the Euclidean distance between E_1 and E_2"

    IMPORTANT: teacher_hidden should be detached (.detach()) before
    passing to this function — we don't want gradients flowing back
    through the teacher.

    If dimensions differ (student_hidden_dim ≠ teacher_d_model), a
    linear projection must be applied beforehand by the caller.

    Args:
        student_hidden: (B, T, student_dim) — student's hidden states E_2.
        teacher_hidden: (B, T, teacher_dim) — teacher's hidden states E_1
                        (detached, no gradients).

    Returns:
        Scalar MSE loss tensor.
    """
    loss_fn = nn.MSELoss()
    return loss_fn(student_hidden, teacher_hidden)


def kl_divergence_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = None,
) -> torch.Tensor:
    """
    KL-Divergence Loss for distribution-level KD (L_DKL in the paper).

    Computes how different the student's output probability distribution
    is from the teacher's. Higher temperature → softer distributions →
    more information about inter-class relationships transferred.

    The paper states:
      "KL-divergence captures the discrepancy between the probabilistic
       distributions of E_1 and E_2. Including KL-divergence enables the
       student to better mimic the distribution of the teacher's internal
       representations."

    Formula:
        L_DKL = KL( softmax(student_logits / T) || softmax(teacher_logits / T) )

    The loss is scaled by T² to make the gradient magnitude consistent
    with the cross-entropy loss (standard practice from Hinton et al., 2015).

    Args:
        student_logits: (B, T, vocab_size) — student's raw outputs.
        teacher_logits: (B, T, vocab_size) — teacher's raw outputs (detached).
        temperature:    Softening temperature (default from config).

    Returns:
        Scalar KL-divergence loss tensor.
    """
    T = temperature or config.kl_temperature

    # Shift both to align predictions
    student_shifted = student_logits[:, :-1, :].contiguous()
    teacher_shifted = teacher_logits[:, :-1, :].contiguous()

    # Soften distributions with temperature
    student_log_probs = F.log_softmax(student_shifted / T, dim=-1)
    teacher_probs = F.softmax(teacher_shifted / T, dim=-1)

    # KL(P || Q) where P = teacher, Q = student
    # PyTorch KLDivLoss expects log(Q) and P
    loss_fn = nn.KLDivLoss(reduction="batchmean")
    kl = loss_fn(student_log_probs, teacher_probs)

    # Scale by T² (Hinton et al., 2015)
    return kl * (T ** 2)


def moscnet_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = None,
    beta: float = None,
    gamma: float = None,
) -> dict:
    """
    Combined MoScNet Knowledge Distillation loss.

    L = α · L_CE^S  +  β · L_L2  +  γ · L_DKL

    Returns individual loss components AND the combined total for logging.

    Args:
        student_logits:  (B, T, vocab_size)
        teacher_logits:  (B, T, vocab_size) — detached
        student_hidden:  (B, T, dim) — student's last hidden states
        teacher_hidden:  (B, T, dim) — teacher's last hidden states (detached)
        labels:          (B, T) — ground truth token IDs
        alpha, beta, gamma: Loss weights (defaults from config)

    Returns:
        Dict with keys: "total", "ce", "l2", "kl" — all scalar tensors.
    """
    a = alpha if alpha is not None else config.alpha_ce
    b = beta if beta is not None else config.beta_l2
    g = gamma if gamma is not None else config.gamma_kl

    # Individual losses
    ce = student_ce_loss(student_logits, labels)
    l2 = l2_loss(student_hidden, teacher_hidden)
    kl = kl_divergence_loss(student_logits, teacher_logits)

    # Combined
    total = a * ce + b * l2 + g * kl

    return {
        "total": total,
        "ce": ce.detach(),
        "l2": l2.detach(),
        "kl": kl.detach(),
    }
