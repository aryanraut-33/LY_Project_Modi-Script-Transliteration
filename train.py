"""
==============================================================================
train.py — Two-Phase Training Orchestrator for MoScNet
==============================================================================

PURPOSE:
    The main entry point that wires all modules together and executes the
    complete MoScNet training pipeline in two phases:

    ╔═══════════════════════════════════════════════════════════════════════╗
    ║  PHASE 1 — Teacher LoRA Fine-Tuning                                ║
    ║                                                                     ║
    ║  Goal: Teach the teacher LLM the Modi→Devanagari transliteration   ║
    ║        task by training only the LoRA adapter weights.             ║
    ║                                                                     ║
    ║  for each epoch:                                                    ║
    ║    for each batch (image, text):                                    ║
    ║      Z_I = vision_encoder(preprocess(image))                       ║
    ║      Z_L = language_encoder.embed(tokenize(text))                  ║
    ║      Z_C = concat(Z_I, Z_L)                                       ║
    ║      logits, _ = teacher(Z_C)                                      ║
    ║      loss = cross_entropy(logits, labels)                          ║
    ║      loss.backward()  → only LoRA params update                    ║
    ╠═══════════════════════════════════════════════════════════════════════╣
    ║  PHASE 2 — Student Knowledge Distillation                          ║
    ║                                                                     ║
    ║  Goal: Train the lightweight student to mimic the teacher's         ║
    ║        outputs using three loss signals.                           ║
    ║                                                                     ║
    ║  teacher.freeze()  → no more gradient updates to teacher           ║
    ║  for each epoch:                                                    ║
    ║    for each batch:                                                  ║
    ║      Z_C = fuse(vision_encoder(image), embed(text))                ║
    ║      teacher_logits, E_1 = teacher(Z_C)   [no_grad]               ║
    ║      student_logits, E_2 = student(Z_C)                            ║
    ║      loss = α·CE + β·L2(E_2, E_1) + γ·KL(student, teacher)       ║
    ║      loss.backward()  → student + vision projector update          ║
    ╚═══════════════════════════════════════════════════════════════════════╝

WHERE IT IS USED:
    This is the top-level script. Run it directly:
        python train.py

HOW IT INTERACTS WITH OTHERS:
    train.py imports and orchestrates EVERY other module:

    ┌────────────────────────────────────────────────────────────────────┐
    │  config.py         → all hyperparameters                          │
    │  utils.py          → seed, logging, checkpointing, param counting│
    │  data_loader.py    → get_dataloaders() for train/val/test         │
    │  preprocessing.py  → (used internally by data_loader)            │
    │  vision_encoder.py → encode Modi images → Z_I                    │
    │  language_encoder.py → tokenize + embed → Z_L, fuse → Z_C       │
    │  teacher_model.py  → Phase 1 training + Phase 2 frozen inference │
    │  student_model.py  → Phase 2 training                            │
    │  losses.py         → teacher_ce_loss, moscnet_kd_loss            │
    │  metrics.py        → compute_bleu for validation                 │
    └────────────────────────────────────────────────────────────────────┘

HARDWARE OPTIMIZATIONS (Colab T4, 15GB VRAM):
    - FP16 Mixed Precision via torch.cuda.amp (T4 has no BF16)
    - Gradient Accumulation: batch_size=1, accum_steps=32 → effective=32
    - Gradient Checkpointing on teacher model
    - Explicit torch.cuda.empty_cache() between phases
    - Teacher frozen with no_grad context in Phase 2
==============================================================================
"""

import os
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from config import config
from utils import set_seed, get_device, setup_logging, save_checkpoint, count_parameters
from data_loader import get_dataloaders
from vision_encoder import MoScNetVisionEncoder
from language_encoder import MoScNetLanguageEncoder
from teacher_model import MoScNetTeacher
from student_model import MoScNetStudent
from losses import teacher_ce_loss, moscnet_kd_loss
from metrics import compute_bleu


def train_teacher_phase(
    teacher: MoScNetTeacher,
    vision_encoder: MoScNetVisionEncoder,
    language_encoder: MoScNetLanguageEncoder,
    train_loader,
    val_loader,
    logger,
):
    """
    Phase 1: Fine-tune the teacher LLM's LoRA adapters.

    Only LoRA parameters are updated. The base LLM, vision encoder,
    and embedding layer are all frozen.
    """
    logger.info("=" * 60)
    logger.info("PHASE 1: Teacher LoRA Fine-Tuning")
    logger.info("=" * 60)

    # Optimizer — only teacher LoRA + vision projector params
    trainable_params = [
        {"params": teacher.model.parameters(), "lr": config.lr_teacher},
        {"params": vision_encoder.projector.parameters(), "lr": config.lr_student},
    ]
    optimizer = torch.optim.AdamW(trainable_params, weight_decay=config.weight_decay)

    # Learning rate scheduler
    total_steps = len(train_loader) * config.epochs_teacher // config.grad_accum_steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # Mixed precision scaler (FP16 for T4)
    scaler = GradScaler(enabled=config.use_fp16)

    best_val_loss = float("inf")

    for epoch in range(config.epochs_teacher):
        teacher.model.train()
        vision_encoder.train()
        epoch_loss = 0.0
        num_batches = 0

        progress = tqdm(train_loader, desc=f"Phase1 Epoch {epoch+1}/{config.epochs_teacher}")

        for step, batch in enumerate(progress):
            # Move to device
            pixel_values = batch["pixel_values"].to(config.device)
            input_ids = batch["input_ids"].to(config.device)
            labels = batch["labels"].to(config.device)

            with autocast(enabled=config.use_fp16):
                # Vision encoding → Z_I
                Z_I = vision_encoder(pixel_values)

                # Language embedding → Z_L
                Z_L = language_encoder(input_ids)

                # Fuse → Z_C
                Z_C = language_encoder.fuse(Z_I, Z_L)

                # Build attention mask for fused sequence
                num_vision = Z_I.size(1)
                num_lang = Z_L.size(1)
                total_len = num_vision + num_lang
                attn_mask = torch.ones(
                    (Z_C.size(0), total_len), device=config.device
                )

                # Teacher forward
                logits, _ = teacher(inputs_embeds=Z_C, attention_mask=attn_mask)

                # Extract only language portion logits for loss
                lang_logits = logits[:, num_vision:, :]

                # Loss
                loss = teacher_ce_loss(lang_logits, labels)
                loss = loss / config.grad_accum_steps

            # Backward with gradient scaling
            scaler.scale(loss).backward()

            # Gradient accumulation
            if (step + 1) % config.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    teacher.model.parameters(), config.max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            epoch_loss += loss.item() * config.grad_accum_steps
            num_batches += 1
            progress.set_postfix({"loss": f"{epoch_loss/num_batches:.4f}"})

        avg_loss = epoch_loss / max(num_batches, 1)
        logger.info(f"Phase1 Epoch {epoch+1} — Train Loss: {avg_loss:.4f}")

        # Validation
        val_loss = validate_teacher(
            teacher, vision_encoder, language_encoder, val_loader
        )
        logger.info(f"Phase1 Epoch {epoch+1} — Val Loss: {val_loss:.4f}")

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                teacher.model, optimizer, epoch, val_loss,
                os.path.join(config.checkpoint_dir, "teacher_best.pt"),
                extra_metadata={"phase": 1, "lora_r": config.lora_r},
            )
            logger.info(f"  → Saved best teacher checkpoint (val_loss={val_loss:.4f})")

    return teacher


@torch.no_grad()
def validate_teacher(teacher, vision_encoder, language_encoder, val_loader):
    """Quick validation loss for the teacher."""
    teacher.model.eval()
    vision_encoder.eval()
    total_loss = 0.0
    count = 0

    for batch in val_loader:
        pixel_values = batch["pixel_values"].to(config.device)
        input_ids = batch["input_ids"].to(config.device)
        labels = batch["labels"].to(config.device)

        with autocast(enabled=config.use_fp16):
            Z_I = vision_encoder(pixel_values)
            Z_L = language_encoder(input_ids)
            Z_C = language_encoder.fuse(Z_I, Z_L)

            num_vision = Z_I.size(1)
            attn_mask = torch.ones(
                (Z_C.size(0), Z_C.size(1)), device=config.device
            )

            logits, _ = teacher(inputs_embeds=Z_C, attention_mask=attn_mask)
            lang_logits = logits[:, num_vision:, :]
            loss = teacher_ce_loss(lang_logits, labels)

        total_loss += loss.item()
        count += 1

    return total_loss / max(count, 1)


def train_student_phase(
    student: MoScNetStudent,
    teacher: MoScNetTeacher,
    vision_encoder: MoScNetVisionEncoder,
    language_encoder: MoScNetLanguageEncoder,
    train_loader,
    val_loader,
    logger,
):
    """
    Phase 2: Train the student via Knowledge Distillation.

    The teacher is completely frozen. The student learns from three signals:
      - Cross-entropy with ground truth (primary task)
      - L2 distance to teacher hidden states (feature alignment)
      - KL-divergence with teacher output distribution (soft targets)
    """
    logger.info("=" * 60)
    logger.info("PHASE 2: Student Knowledge Distillation")
    logger.info("=" * 60)

    # Freeze teacher
    teacher.freeze()

    # Optimizer — student + vision projector
    trainable_params = [
        {"params": student.parameters(), "lr": config.lr_student},
        {"params": vision_encoder.projector.parameters(), "lr": config.lr_student},
    ]
    optimizer = torch.optim.AdamW(trainable_params, weight_decay=config.weight_decay)

    total_steps = len(train_loader) * config.epochs_student // config.grad_accum_steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    scaler = GradScaler(enabled=config.use_fp16)

    best_val_bleu = 0.0

    for epoch in range(config.epochs_student):
        student.train()
        vision_encoder.train()
        epoch_losses = {"total": 0, "ce": 0, "l2": 0, "kl": 0}
        num_batches = 0

        progress = tqdm(train_loader, desc=f"Phase2 Epoch {epoch+1}/{config.epochs_student}")

        for step, batch in enumerate(progress):
            pixel_values = batch["pixel_values"].to(config.device)
            input_ids = batch["input_ids"].to(config.device)
            labels = batch["labels"].to(config.device)

            with autocast(enabled=config.use_fp16):
                # Shared encoding
                Z_I = vision_encoder(pixel_values)
                Z_L = language_encoder(input_ids)
                Z_C = language_encoder.fuse(Z_I, Z_L)

                num_vision = Z_I.size(1)
                num_lang = Z_L.size(1)

                # Causal mask
                mask = language_encoder.create_causal_mask(
                    num_vision, num_lang, device=config.device
                )

                # Teacher forward (frozen, no gradients)
                with torch.no_grad():
                    attn_mask = torch.ones(
                        (Z_C.size(0), Z_C.size(1)), device=config.device
                    )
                    teacher_logits, teacher_hidden = teacher(
                        inputs_embeds=Z_C, attention_mask=attn_mask
                    )
                    # Extract language portion only
                    teacher_lang_logits = teacher_logits[:, num_vision:, :]
                    teacher_lang_hidden = teacher_hidden[:, num_vision:, :]

                # Student forward
                student_logits, student_hidden = student(Z_C, mask)
                # Extract language portion only
                student_lang_logits = student_logits[:, num_vision:, :]
                student_lang_hidden = student_hidden[:, num_vision:, :]

                # Combined KD loss
                loss_dict = moscnet_kd_loss(
                    student_logits=student_lang_logits,
                    teacher_logits=teacher_lang_logits.detach(),
                    student_hidden=student_lang_hidden,
                    teacher_hidden=teacher_lang_hidden.detach(),
                    labels=labels,
                )

                loss = loss_dict["total"] / config.grad_accum_steps

            # Backward
            scaler.scale(loss).backward()

            # Gradient accumulation
            if (step + 1) % config.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(student.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            # Accumulate logging
            for key in epoch_losses:
                epoch_losses[key] += loss_dict[key].item() if key != "total" else loss_dict["total"].item()
            num_batches += 1

            progress.set_postfix({
                "loss": f"{epoch_losses['total']/num_batches:.4f}",
                "ce": f"{epoch_losses['ce']/num_batches:.4f}",
            })

        # Log epoch stats
        for key in epoch_losses:
            epoch_losses[key] /= max(num_batches, 1)
        logger.info(
            f"Phase2 Epoch {epoch+1} — "
            f"Total: {epoch_losses['total']:.4f}, "
            f"CE: {epoch_losses['ce']:.4f}, "
            f"L2: {epoch_losses['l2']:.4f}, "
            f"KL: {epoch_losses['kl']:.4f}"
        )

        # Validation BLEU
        val_bleu = validate_student(
            student, vision_encoder, language_encoder, val_loader
        )
        logger.info(f"Phase2 Epoch {epoch+1} — Val BLEU: {val_bleu:.2f}")

        # Save best model
        if val_bleu > best_val_bleu:
            best_val_bleu = val_bleu
            save_checkpoint(
                student, optimizer, epoch, val_bleu,
                os.path.join(config.checkpoint_dir, "student_best.pt"),
                extra_metadata={"variant": config.student_variant},
            )
            logger.info(f"  → Saved best student (BLEU={val_bleu:.2f})")

    return student


@torch.no_grad()
def validate_student(student, vision_encoder, language_encoder, val_loader):
    """
    Compute BLEU score on the validation set using greedy decoding.
    """
    student.eval()
    vision_encoder.eval()

    all_predictions = []
    all_references = []
    tokenizer = language_encoder.tokenizer

    for batch in val_loader:
        pixel_values = batch["pixel_values"].to(config.device)
        input_ids = batch["input_ids"].to(config.device)
        labels = batch["labels"]

        with autocast(enabled=config.use_fp16):
            Z_I = vision_encoder(pixel_values)
            Z_L = language_encoder(input_ids)
            Z_C = language_encoder.fuse(Z_I, Z_L)

            num_vision = Z_I.size(1)
            num_lang = Z_L.size(1)
            mask = language_encoder.create_causal_mask(
                num_vision, num_lang, device=config.device
            )

            logits, _ = student(Z_C, mask)
            lang_logits = logits[:, num_vision:, :]

        # Greedy decoding
        pred_ids = lang_logits.argmax(dim=-1)

        # Decode to strings
        for i in range(pred_ids.size(0)):
            pred_text = tokenizer.decode(pred_ids[i], skip_special_tokens=True)
            # Reconstruct reference from labels (replace -100 with pad)
            ref_ids = labels[i].clone()
            ref_ids[ref_ids == -100] = tokenizer.pad_token_id
            ref_text = tokenizer.decode(ref_ids, skip_special_tokens=True)

            all_predictions.append(pred_text)
            all_references.append(ref_text)

    if not all_predictions:
        return 0.0

    return compute_bleu(all_predictions, all_references)


def main():
    """
    Main training entry point.
    """
    # ── Setup ────────────────────────────────────────────────────────
    set_seed(config.seed)
    device = get_device()
    config.device = device
    logger = setup_logging(config.log_dir)

    logger.info(f"Device: {device}")
    logger.info(f"Student variant: {config.student_variant}")
    logger.info(f"Teacher: {config.teacher_model_name}")

    # Create output directories
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.output_dir, exist_ok=True)

    # ── Initialize Models ────────────────────────────────────────────
    logger.info("Initializing models...")

    # Language encoder (loads tokenizer)
    language_encoder = MoScNetLanguageEncoder()
    tokenizer = language_encoder.tokenizer

    # Vision encoder (loads SigLIP, freezes it)
    vision_encoder = MoScNetVisionEncoder().to(device)
    logger.info(f"Vision encoder params: {count_parameters(vision_encoder)}")

    # Teacher (loads 4-bit LLaMA + LoRA)
    teacher = MoScNetTeacher()
    # Initialize language encoder embeddings from teacher
    language_encoder.init_from_teacher(teacher.get_embedding_weight())
    language_encoder = language_encoder.to(device)

    # Student model
    student = MoScNetStudent(vocab_size=len(tokenizer)).to(device)
    logger.info(f"Student params: {count_parameters(student)}")

    # ── Data ─────────────────────────────────────────────────────────
    logger.info("Loading data...")
    loaders = get_dataloaders(tokenizer)

    # ── Phase 1: Teacher Training ────────────────────────────────────
    teacher = train_teacher_phase(
        teacher, vision_encoder, language_encoder,
        loaders["train"], loaders["val"], logger,
    )

    # Free some VRAM
    torch.cuda.empty_cache()

    # ── Phase 2: Student Training ────────────────────────────────────
    student = train_student_phase(
        student, teacher, vision_encoder, language_encoder,
        loaders["train"], loaders["val"], logger,
    )

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
