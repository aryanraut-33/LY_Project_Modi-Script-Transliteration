"""
==============================================================================
metrics.py — Evaluation Metrics for Modi-to-Devanagari Transliteration
==============================================================================

PURPOSE:
    Computes quantitative evaluation metrics to measure how well the student
    model transliterates Modi script images into Devanagari text.

    Three metrics are implemented:

    1. BLEU Score (Bilingual Evaluation Understudy)
       ─ The PAPER'S PRIMARY METRIC (Section 5, Tables 1–6).
       ─ Measures n-gram overlap between predicted and reference text.
       ─ Target: 51.03 on MoDeTrans (paper's best with MoScNet-XL).
       ─ Implemented via sacrebleu for standardized, reproducible scores.

    2. Character Error Rate (CER)
       ─ NOT in the paper — added for deeper diagnostics.
       ─ Levenshtein distance at character level / total reference characters.
       ─ Reveals single-character confusions (e.g., pha ↔ ka, bha ↔ ma).

    3. Word Error Rate (WER)
       ─ NOT in the paper — added for word-level insight.
       ─ Levenshtein distance at word level / total reference words.
       ─ In Modi script (no word boundaries), this metric is less meaningful
         but still useful for comparison with other OCR systems.

WHERE IT IS USED:
    - train.py    → compute_bleu() is called at the end of each validation
                    epoch to track the primary training signal and decide
                    whether to save a "best model" checkpoint
    - evaluate.py → all three metrics are computed on the test set for the
                    final evaluation report

HOW IT INTERACTS WITH OTHERS:
    metrics.py is a pure utility module with NO project dependencies.
    It receives plain Python strings (predictions and references) and
    returns float scores. The decoding from token IDs → strings happens
    in evaluate.py before calling these functions.

    Data flow:
        student_model → logits → argmax/beam_search → token IDs
              ↓
        tokenizer.decode() → predicted strings
              ↓
        metrics.py(predictions, references) → BLEU / CER / WER
==============================================================================
"""

import sacrebleu
from typing import List


def compute_bleu(predictions: List[str], references: List[str]) -> float:
    """
    Compute corpus-level BLEU score using sacrebleu.

    This is the paper's primary evaluation metric (Section 5, Appendix A.3).
    BLEU measures n-gram precision (1-gram through 4-gram) with a brevity
    penalty to discourage overly short translations.

    Formula (from Papineni et al., 2002):
        BLEU = BP × exp( Σ wₙ log pₙ )
    where pₙ is n-gram precision and BP penalizes short outputs.

    Args:
        predictions: List of predicted Devanagari strings (one per image).
        references:  List of ground-truth Devanagari strings (one per image).

    Returns:
        BLEU score as a float (0–100 scale).

    Example:
        >>> compute_bleu(["नमस्ते दुनिया"], ["नमस्ते दुनिया"])
        100.0
    """
    # sacrebleu expects references as a list of lists (supporting multiple refs)
    refs_wrapped = [references]
    bleu = sacrebleu.corpus_bleu(predictions, refs_wrapped)
    return bleu.score


def compute_cer(predictions: List[str], references: List[str]) -> float:
    """
    Compute Character Error Rate (CER).

    CER = (total edit distance at character level) / (total reference chars) × 100

    This is useful for diagnosing fine-grained confusions between visually
    similar Modi characters (e.g., bha ↔ ma, ka ↔ pha, as noted in
    Section 5.1 of the paper).

    Args:
        predictions: List of predicted strings.
        references:  List of reference strings.

    Returns:
        CER as a percentage (0–100+, can exceed 100 if predictions are
        much longer than references).

    Note:
        Requires the `editdistance` package. Falls back to a simple
        implementation if not installed.
    """
    try:
        import editdistance
    except ImportError:
        raise ImportError(
            "CER computation requires `editdistance`. "
            "Install it: pip install editdistance"
        )

    total_errors = 0
    total_chars = 0
    for pred, ref in zip(predictions, references):
        total_errors += editdistance.eval(pred, ref)
        total_chars += len(ref)

    if total_chars == 0:
        return 0.0
    return (total_errors / total_chars) * 100


def compute_wer(predictions: List[str], references: List[str]) -> float:
    """
    Compute Word Error Rate (WER).

    WER = (total edit distance at word level) / (total reference words) × 100

    Note: Modi script has no spaces between words, so WER is less meaningful
    for this task. The Devanagari transliterations DO have word boundaries,
    so this metric is computed on the output side.

    Args:
        predictions: List of predicted strings.
        references:  List of reference strings.

    Returns:
        WER as a percentage (0–100+).
    """
    try:
        import editdistance
    except ImportError:
        raise ImportError(
            "WER computation requires `editdistance`. "
            "Install it: pip install editdistance"
        )

    total_errors = 0
    total_words = 0
    for pred, ref in zip(predictions, references):
        pred_words = pred.split()
        ref_words = ref.split()
        total_errors += editdistance.eval(pred_words, ref_words)
        total_words += len(ref_words)

    if total_words == 0:
        return 0.0
    return (total_errors / total_words) * 100
