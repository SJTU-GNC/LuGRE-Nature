from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def compare_png(generated: Path, reference: Path) -> dict[str, object]:
    if not generated.is_file():
        raise FileNotFoundError(f"Generated PNG is missing: {generated}")
    if not reference.is_file():
        raise FileNotFoundError(f"Reference PNG is missing: {reference}")

    generated_hash = sha256(generated)
    reference_hash = sha256(reference)
    with Image.open(generated) as left_image, Image.open(reference) as right_image:
        left = np.asarray(left_image.convert("RGBA"), dtype=np.int16)
        right = np.asarray(right_image.convert("RGBA"), dtype=np.int16)

    same_shape = left.shape == right.shape
    if same_shape:
        delta = np.abs(left - right)
        different_pixels = int(np.any(delta, axis=2).sum())
        max_channel_delta = int(delta.max(initial=0))
        mean_channel_delta = float(delta.mean())
    else:
        different_pixels = -1
        max_channel_delta = -1
        mean_channel_delta = float("nan")

    return {
        "generated": str(generated),
        "reference": str(reference),
        "generated_sha256": generated_hash,
        "reference_sha256": reference_hash,
        "byte_identical": generated_hash == reference_hash,
        "same_shape": same_shape,
        "different_pixels": different_pixels,
        "max_channel_delta": max_channel_delta,
        "mean_channel_delta": mean_channel_delta,
    }
