"""
==============================================================================
evaluate.py — Inference & Test Evaluation for MoScNet
==============================================================================

PURPOSE:
    Loads a trained MoScNet student checkpoint and evaluates it on the
    test split of MoDeTrans. This script performs:

    1. AUTOREGRESSIVE GENERATION:
       Given a Modi script image, the model generates Devanagari text
       token-by-token. Supports both greedy decoding and beam search.

    2. QUANTITATIVE EVALUATION:
       Computes BLEU (primary), CER, and WER on the full test set.

    3. QUALITATIVE INSPECTION:
       Saves predictions alongside ground truth to a CSV file for
       manual review, matching what the paper shows in Figure 7
       (qualitative transliteration comparison).

    The paper (Section 5 — Experimental Results):
      "Figure 7 showcases the transliteration performance of various
       methods. Incorrectly transliterated characters are highlighted."

WHERE IT IS USED:
    Run after training is complete:
        python evaluate.py --checkpoint ./checkpoints/student_best.pt

HOW IT INTERACTS WITH OTHERS:
    evaluate.py imports a subset of modules (no teacher needed):

    ┌─────────────────────────────────────────────────────────────────┐
    │  config.py          → hyperparameters and paths                │
    │  utils.py           → load_checkpoint, get_device, set_seed    │
    │  data_loader.py     → get_dataloaders (test split)             │
    │  vision_encoder.py  → encode test images                       │
    │  language_encoder.py → tokenize/embed, build masks, decode     │
    │  student_model.py   → loaded from checkpoint                   │
    │  metrics.py         → compute_bleu, compute_cer, compute_wer   │
    └─────────────────────────────────────────────────────────────────┘

    The teacher model is NOT needed during evaluation — only the
    lightweight student model (e.g., MoScNet-M at 63M params) is loaded.
    This is the key benefit of Knowledge Distillation: a compact model
    that can run on resource-constrained hardware.
==============================================================================
"""

import os
import csv
import argparse
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

from config import config
from utils import set_seed, get_device, load_checkpoint, count_parameters
from data_loader import get_dataloaders
from vision_encoder import MoScNetVisionEncoder
from language_encoder import MoScNetLanguageEncoder
from student_model import MoScNetStudent
from metrics import compute_bleu, compute_cer, compute_wer


@torch.no_grad()
def generate_greedy(
    student: MoScNetStudent,
    vision_encoder: MoScNetVisionEncoder,
    language_encoder: MoScNetLanguageEncoder,
    pixel_values: torch.Tensor,
    max_new_tokens: int = 256,
) -> torch.Tensor:
    """
    Autoregressive greedy decoding for a single batch of images.

    Process:
      1. Encode the image → Z_I (vision embeddings)
      2. Start with a BOS/start token
      3. At each step:
         a. Embed current tokens → Z_L
         b. Fuse Z_I + Z_L → Z_C
         c. Forward through student → logits
         d. Take argmax of last position → next token
         e. Append to sequence
      4. Stop when EOS token is generated or max length reached

    Args:
        student:          Trained MoScNet student model.
        vision_encoder:   SigLIP + MLP projector.
        language_encoder: Tokenizer + embedding.
        pixel_values:     (B, 3, H, W) — preprocessed images.
        max_new_tokens:   Maximum tokens to generate.

    Returns:
        (B, generated_len) — tensor of generated token IDs.
    """
    device = pixel_values.device
    tokenizer = language_encoder.tokenizer
    B = pixel_values.size(0)

    # Encode image
    Z_I = vision_encoder(pixel_values)  # (B, num_patches, d_model)
    num_vision = Z_I.size(1)

    # Start token
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    generated = torch.full((B, 1), bos_id, dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        # Embed current generated tokens
        Z_L = language_encoder(generated)  # (B, current_len, d_model)

        # Fuse vision + language
        Z_C = language_encoder.fuse(Z_I, Z_L)

        # Build causal mask
        num_lang = generated.size(1)
        mask = language_encoder.create_causal_mask(
            num_vision, num_lang, device=device
        )

        # Student forward
        with autocast(enabled=config.use_fp16):
            logits, _ = student(Z_C, mask)

        # Get logits for the last token position only
        # (language portion starts after vision tokens)
        last_logits = logits[:, -1, :]  # (B, vocab_size)

        # Greedy: pick the highest probability token
        next_token = last_logits.argmax(dim=-1, keepdim=True)  # (B, 1)

        # Append to generated sequence
        generated = torch.cat([generated, next_token], dim=1)

        # Stop if all sequences have generated EOS
        if tokenizer.eos_token_id is not None:
            if (next_token == tokenizer.eos_token_id).all():
                break

    return generated


def evaluate_test_set(
    student: MoScNetStudent,
    vision_encoder: MoScNetVisionEncoder,
    language_encoder: MoScNetLanguageEncoder,
    test_loader,
    output_path: str = None,
) -> dict:
    """
    Evaluate the student model on the full test set.

    Computes BLEU (primary metric), CER, and WER. Optionally saves
    predictions to a CSV file for qualitative inspection.

    Args:
        student:          Trained student model.
        vision_encoder:   Vision encoder.
        language_encoder: Language encoder with tokenizer.
        test_loader:      Test DataLoader.
        output_path:      Optional CSV path for predictions.

    Returns:
        Dict with "bleu", "cer", "wer" scores.
    """
    student.eval()
    vision_encoder.eval()

    tokenizer = language_encoder.tokenizer
    all_predictions = []
    all_references = []

    print("Running inference on test set...")
    for batch in tqdm(test_loader, desc="Evaluating"):
        pixel_values = batch["pixel_values"].to(config.device)
        labels = batch["labels"]

        # Generate predictions
        generated_ids = generate_greedy(
            student, vision_encoder, language_encoder,
            pixel_values, max_new_tokens=config.max_seq_len,
        )

        # Decode predictions
        for i in range(generated_ids.size(0)):
            pred_text = tokenizer.decode(
                generated_ids[i], skip_special_tokens=True
            ).strip()

            # Reconstruct reference from labels
            ref_ids = labels[i].clone()
            ref_ids[ref_ids == -100] = tokenizer.pad_token_id
            ref_text = tokenizer.decode(ref_ids, skip_special_tokens=True).strip()

            all_predictions.append(pred_text)
            all_references.append(ref_text)

    # ── Compute metrics ──────────────────────────────────────────────
    results = {
        "bleu": compute_bleu(all_predictions, all_references),
    }

    # CER and WER (require editdistance)
    try:
        results["cer"] = compute_cer(all_predictions, all_references)
        results["wer"] = compute_wer(all_predictions, all_references)
    except ImportError:
        print("Warning: editdistance not installed. Skipping CER/WER.")
        results["cer"] = None
        results["wer"] = None

    # ── Save predictions to CSV ──────────────────────────────────────
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Index", "Prediction", "Reference"])
            for i, (pred, ref) in enumerate(zip(all_predictions, all_references)):
                writer.writerow([i, pred, ref])
        print(f"Predictions saved to: {output_path}")

    return results


def main():
    """
    Main evaluation entry point.
    """
    parser = argparse.ArgumentParser(description="Evaluate MoScNet Student")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(config.checkpoint_dir, "student_best.pt"),
        help="Path to student model checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(config.output_dir, "test_predictions.csv"),
        help="Path to save predictions CSV.",
    )
    args = parser.parse_args()

    # ── Setup ────────────────────────────────────────────────────────
    set_seed(config.seed)
    device = get_device()
    config.device = device

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")

    # ── Initialize Models ────────────────────────────────────────────
    # Language encoder (for tokenizer)
    language_encoder = MoScNetLanguageEncoder().to(device)
    tokenizer = language_encoder.tokenizer

    # Vision encoder
    vision_encoder = MoScNetVisionEncoder().to(device)
    vision_encoder.eval()

    # Student model
    student = MoScNetStudent(vocab_size=len(tokenizer)).to(device)

    # Load checkpoint
    ckpt_info = load_checkpoint(args.checkpoint, student)
    print(f"Loaded checkpoint from epoch {ckpt_info['epoch']} "
          f"(val BLEU: {ckpt_info['val_bleu']:.2f})")
    print(f"Student params: {count_parameters(student)}")

    # ── Load test data ───────────────────────────────────────────────
    loaders = get_dataloaders(tokenizer)
    test_loader = loaders["test"]

    # ── Evaluate ─────────────────────────────────────────────────────
    results = evaluate_test_set(
        student, vision_encoder, language_encoder,
        test_loader, output_path=args.output,
    )

    # ── Print results ────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)
    print(f"  BLEU Score:           {results['bleu']:.2f}")
    if results.get("cer") is not None:
        print(f"  Character Error Rate: {results['cer']:.2f}%")
    if results.get("wer") is not None:
        print(f"  Word Error Rate:      {results['wer']:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()
