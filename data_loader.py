"""
==============================================================================
data_loader.py — Dataset Loading & DataLoader Construction for MoScNet
==============================================================================

PURPOSE:
    Downloads the MoDeTrans dataset from HuggingFace Hub, splits it into
    train/val/test sets (80:10:10 as per Section 3 of the paper), wraps
    each split in a PyTorch Dataset that applies image preprocessing and
    text tokenization, and returns ready-to-iterate DataLoaders.

    The MoDeTrans dataset contains 2,043 images of Modi script documents
    with their corresponding Devanagari transliterations. Each sample is
    a dict with:
        - "image": PIL.Image  (preprocessed Modi script document patch)
        - "text":  str         (Devanagari transliteration)

WHERE IT IS USED:
    - train.py    → `get_dataloaders(tokenizer)` is called once at startup
                    to obtain train_loader, val_loader, and test_loader.
    - evaluate.py → uses the test_loader for final evaluation.

HOW IT INTERACTS WITH OTHERS:
    data_loader.py is the bridge between raw data and the model pipeline:

    1. Imports MoScNetPreprocessor from preprocessing.py to transform images
       and clean text.
    2. Receives a tokenizer (from language_encoder.py or teacher_model.py)
       to convert cleaned Devanagari text into token IDs.
    3. Outputs batches of dicts with keys:
         - "pixel_values":   (B, 3, 224, 224)  → fed to vision_encoder.py
         - "input_ids":      (B, max_seq_len)   → fed to language_encoder.py
         - "attention_mask": (B, max_seq_len)   → fed to teacher/student
         - "labels":         (B, max_seq_len)   → fed to losses.py

    Dependency chain:
        HuggingFace Hub
              ↓
        data_loader.py  ←──  preprocessing.py (transforms)
              │               config.py (dataset name, batch size, seed)
              ↓
        DataLoader batches  ──→  train.py / evaluate.py
==============================================================================
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from typing import Dict, Any, List, Optional
from PIL import Image

from config import config
from preprocessing import MoScNetPreprocessor


class MoDeTransDataset(Dataset):
    """
    PyTorch Dataset wrapper for the MoDeTrans HuggingFace dataset.

    Each __getitem__ call:
      1. Loads a (Modi image, Devanagari text) pair
      2. Preprocesses the image (resize, normalize, optional augmentation)
      3. Cleans the text (remove non-Devanagari characters)
      4. Tokenizes the text using the teacher LLM's BPE tokenizer
      5. Returns a dict of tensors ready for the model

    Args:
        data:         A list-like of dicts with keys "image" and "text".
        preprocessor: MoScNetPreprocessor instance for image/text transforms.
        tokenizer:    HuggingFace tokenizer (from the teacher LLM).
        is_train:     If True, applies data augmentation to images.
    """

    def __init__(
        self,
        data: Any,
        preprocessor: MoScNetPreprocessor,
        tokenizer: Any,
        is_train: bool = True,
    ):
        self.data = data
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]
        image = sample["image"]
        text = sample["text"]

        # ── Image preprocessing ──────────────────────────────────────
        # Input:  PIL image (already binarized/denoised by dataset creators)
        # Output: Tensor (3, 224, 224) normalized for SigLIP
        pixel_values = self.preprocessor.preprocess_image(
            image, is_train=self.is_train
        )

        # ── Text preprocessing & tokenization ────────────────────────
        # Step 1: Clean (remove non-Devanagari chars, normalize whitespace)
        cleaned_text = self.preprocessor.preprocess_text(text)

        # Step 2: Tokenize with the teacher's BPE tokenizer
        # The paper reuses the LLM tokenizer without modification because
        # LLaMA-3 was trained on Hindi data (same Devanagari script)
        tokenized = self.tokenizer(
            cleaned_text,
            max_length=config.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = tokenized["input_ids"].squeeze(0)       # (max_seq_len,)
        attention_mask = tokenized["attention_mask"].squeeze(0)  # (max_seq_len,)

        # ── Labels for causal language modeling ──────────────────────
        # Labels = input_ids, but pad tokens are set to -100 so they
        # are ignored by CrossEntropyLoss. The loss function internally
        # handles the shift (predict token t+1 from token t).
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,      # (3, 224, 224)
            "input_ids": input_ids,            # (max_seq_len,)
            "attention_mask": attention_mask,   # (max_seq_len,)
            "labels": labels,                  # (max_seq_len,)
        }


def get_dataloaders(tokenizer: Any) -> Dict[str, DataLoader]:
    """
    Download MoDeTrans, split 80:10:10, and return DataLoaders.

    The paper (Section 3):
      "The dataset was divided into training, testing, and validation
       sets in an 80:10:10 ratio. The images from each era were
       proportionally split across these sets."

    Args:
        tokenizer: HuggingFace tokenizer (passed in from language_encoder
                   or teacher_model initialization).

    Returns:
        Dict with keys "train", "val", "test", each mapping to a DataLoader.
    """
    # ── Download from HuggingFace Hub ────────────────────────────────
    print(f"Loading dataset: {config.dataset_name}")
    dataset = load_dataset(config.dataset_name)

    # The dataset has a single "train" split — we split it manually
    full_data = dataset["train"]

    # ── 80:10:10 split ───────────────────────────────────────────────
    # First split: 80% train, 20% temp
    train_testval = full_data.train_test_split(
        test_size=(1.0 - config.train_split), seed=config.seed
    )
    # Second split: 50% of temp = 10% val, 10% test
    test_val = train_testval["test"].train_test_split(
        test_size=0.5, seed=config.seed
    )

    splits = {
        "train": train_testval["train"],
        "val": test_val["train"],
        "test": test_val["test"],
    }

    print(f"  Train: {len(splits['train'])} samples")
    print(f"  Val:   {len(splits['val'])} samples")
    print(f"  Test:  {len(splits['test'])} samples")

    # ── Build Datasets & DataLoaders ─────────────────────────────────
    preprocessor = MoScNetPreprocessor(img_size=config.img_size)

    loaders = {}
    for split_name, split_data in splits.items():
        is_train = (split_name == "train")
        ds = MoDeTransDataset(
            data=split_data,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            is_train=is_train,
        )
        loaders[split_name] = DataLoader(
            ds,
            batch_size=config.batch_size,
            shuffle=is_train,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=is_train,  # drop incomplete last batch during training
        )

    return loaders
