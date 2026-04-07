"""
==============================================================================
preprocessing.py — Image & Text Preprocessing for MoScNet
==============================================================================

PURPOSE:
    Transforms raw dataset samples into model-ready tensors. This module
    handles TWO separate preprocessing pipelines:

    1. IMAGE PIPELINE:
       The MoDeTrans dataset images are already preprocessed by the paper
       authors (grayscale → bilateral filter → Gaussian adaptive threshold
       → deskew). Our job is to prepare them for the SigLIP vision encoder:
         • Convert to RGB (SigLIP expects 3-channel input)
         • Resize to 224×224 (SigLIP-Base input resolution)
         • Normalize to [−1, 1] using mean=0.5, std=0.5
         • (Training only) Apply light augmentations for robustness

    2. TEXT PIPELINE:
       Clean the Devanagari transliteration text by removing non-Devanagari
       characters and normalizing whitespace, as described in Section 3 of
       the paper: "Special characters and punctuation were then removed."

WHERE IT IS USED:
    - data_loader.py → MoDeTransDataset.__getitem__() calls the preprocessor
                       to transform each (image, text) pair before batching
    - evaluate.py    → uses the validation/test transforms (no augmentation)

HOW IT INTERACTS WITH OTHERS:
    preprocessing.py reads image size and normalization constants from
    config.py. It produces torch.Tensors consumed by vision_encoder.py
    (image) and strings consumed by language_encoder.py (text).

    Data flow:
        HuggingFace dataset sample
              │
              ├── image (PIL) ──→ preprocessing.py ──→ pixel_values (Tensor)
              │                                              ↓
              │                                      vision_encoder.py
              │
              └── text (str)  ──→ preprocessing.py ──→ cleaned text (str)
                                                             ↓
                                                     language_encoder.py
==============================================================================
"""

import torch
import re
from torchvision import transforms
from PIL import Image
from typing import Tuple
from config import config


def get_image_transforms(
    img_size: Tuple[int, int] = None,
    mean: Tuple[float, ...] = None,
    std: Tuple[float, ...] = None,
    is_train: bool = True,
) -> transforms.Compose:
    """
    Build the torchvision transform pipeline for Modi script images.

    Training transforms include light augmentations (rotation, color jitter)
    to improve generalization on the small MoDeTrans dataset (2,043 images).
    Validation/test transforms apply only resize + normalize.

    Args:
        img_size:  Target (H, W). Defaults to config.img_size.
        mean:      Per-channel mean for normalization. Defaults to config.img_mean.
        std:       Per-channel std for normalization. Defaults to config.img_std.
        is_train:  Whether to include augmentations.

    Returns:
        A torchvision.transforms.Compose pipeline.
    """
    img_size = img_size or config.img_size
    mean = mean or config.img_mean
    std = std or config.img_std

    normalize = transforms.Normalize(mean=list(mean), std=list(std))

    if is_train:
        return transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomRotation(degrees=3),           # slight tilt robustness
            transforms.ColorJitter(brightness=0.1, contrast=0.1),  # lighting variation
            transforms.ToTensor(),                           # [0, 255] → [0.0, 1.0]
            normalize,                                       # [0.0, 1.0] → [−1.0, 1.0]
        ])
    else:
        return transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])


def clean_devanagari_text(text: str) -> str:
    """
    Clean Devanagari transliteration text.

    Following Section 3 of the paper:
      "Special characters and punctuation were then removed to ensure
       cleaner and more structured data for further processing."

    This function:
      1. Keeps only Devanagari Unicode characters (U+0900–U+097F) and spaces
      2. Collapses multiple spaces into one
      3. Strips leading/trailing whitespace

    Args:
        text: Raw Devanagari text from the dataset.

    Returns:
        Cleaned text string.
    """
    if not text:
        return ""
    # Keep Devanagari block (vowels, consonants, matras, numerals) + whitespace
    text = re.sub(r'[^\u0900-\u097F\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class MoScNetPreprocessor:
    """
    Stateful preprocessor that holds transform pipelines for train and eval.

    Usage:
        preprocessor = MoScNetPreprocessor()
        pixel_values = preprocessor.preprocess_image(pil_image, is_train=True)
        clean_text   = preprocessor.preprocess_text(raw_text)
    """

    def __init__(self, img_size: Tuple[int, int] = None):
        _img_size = img_size or config.img_size
        self.train_transforms = get_image_transforms(_img_size, is_train=True)
        self.val_transforms = get_image_transforms(_img_size, is_train=False)

    def preprocess_image(self, image: Image.Image, is_train: bool = True) -> torch.Tensor:
        """
        Convert a PIL image to a normalized tensor for SigLIP.

        Steps:
          1. Convert to RGB (SigLIP requires 3-channel input, but MoDeTrans
             images are grayscale/binarized — we replicate into 3 channels)
          2. Resize to (224, 224)
          3. Normalize to SigLIP stats

        Args:
            image:    PIL.Image from the dataset.
            is_train: If True, applies augmentations.

        Returns:
            Tensor of shape (3, H, W).
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        transform = self.train_transforms if is_train else self.val_transforms
        return transform(image)

    def preprocess_text(self, text: str) -> str:
        """
        Clean Devanagari text (delegates to clean_devanagari_text).

        Args:
            text: Raw text string from the dataset.

        Returns:
            Cleaned Devanagari text.
        """
        return clean_devanagari_text(text)
