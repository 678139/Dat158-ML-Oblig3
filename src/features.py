from __future__ import annotations
from typing import List
import numpy as np
from PIL import Image


def _ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def extract_features(image: Image.Image, bins_per_channel: int = 16) -> np.ndarray:
    image = _ensure_rgb(image)
    arr = np.asarray(image)

    features: List[np.ndarray] = []
    for c in range(3):
        hist, _ = np.histogram(arr[..., c], bins=bins_per_channel, range=(0, 256))
        hist = hist.astype("float32")
        s = hist.sum()
        if s > 0:
            hist /= s
        features.append(hist)

    vec = np.concatenate(features, axis=0)
    return vec.astype("float32")
