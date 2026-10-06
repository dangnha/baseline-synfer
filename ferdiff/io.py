"""Small image I/O helpers shared by the CLI scripts."""

from __future__ import annotations

import os

import numpy as np


def load_image(path: str) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def save_image(path: str, arr) -> None:
    from PIL import Image

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    Image.fromarray((np.clip(np.asarray(arr), 0, 1) * 255).astype("uint8")).save(path)


def basename(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0] + ".png"
