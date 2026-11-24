from __future__ import annotations

"""Utilities for the optional 20-image personal-category extension."""

from pathlib import Path
from typing import List, Tuple
import random

from .dataset import ImageRecord


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def load_personal_extension(
    image_dir: str | Path,
    *,
    category: int = 21,
    n_images: int = 20,
    n_queries: int = 5,
    seed: int = 3032,
) -> Tuple[List[ImageRecord], List[ImageRecord]]:
    """
    Load exactly 20 student-acquired images and create a reproducible 15/5 split.

    Returns (personal_gallery, personal_queries).  The split is deterministic for
    a fixed set of filenames and seed, so students can reproduce their experiment.
    """
    folder = Path(image_dir).expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Personal image folder not found: {folder}")

    paths = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
    )
    if len(paths) != n_images:
        raise ValueError(
            f"Expected exactly {n_images} personal images in {folder}, found {len(paths)}."
        )
    if not (1 <= n_queries < n_images):
        raise ValueError("n_queries must be between 1 and n_images - 1.")

    rng = random.Random(seed)
    indices = list(range(n_images))
    rng.shuffle(indices)
    query_idx = set(indices[:n_queries])

    gallery: List[ImageRecord] = []
    queries: List[ImageRecord] = []
    for i, path in enumerate(paths):
        record = ImageRecord(
            path=path,
            filename=path.name,
            category=category,
            split="personal_query" if i in query_idx else "personal_gallery",
        )
        (queries if i in query_idx else gallery).append(record)

    return gallery, queries
