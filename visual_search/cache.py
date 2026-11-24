"""Functions for saving and reusing per-image local-feature descriptors."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from cvpr.dataset import ImageRecord, read_rgb

from .features import create_sift, extract_opponent_sift


def feature_cache_path(cache_folder: str | Path, image_index: int, record: ImageRecord) -> Path:
    """Build the cache filename used throughout the project."""
    return Path(cache_folder) / f"{image_index:03d}_{record.path.stem}.npz"


def save_feature_cache(
    path: str | Path,
    record: ImageRecord,
    descriptors: np.ndarray,
    keypoint_info: np.ndarray | None = None,
) -> None:
    """Save descriptors and optional keypoint locations in a compressed file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(descriptors, dtype=np.float32)
    payload = {
        "descriptors": values,
        "filename": np.asarray(record.filename),
        "category": np.asarray(record.category),
    }
    if keypoint_info is not None:
        payload["keypoint_info"] = np.asarray(keypoint_info, dtype=np.float32)
    np.savez_compressed(path, **payload)


def load_feature_cache(
    path: str | Path,
    expected_filename: str | None = None,
    expected_dimension: int | None = None,
    include_keypoints: bool = False,
):
    """Load a descriptor cache and optionally its saved keypoint properties."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Feature cache not found: {path}")

    with np.load(path, allow_pickle=False) as saved:
        descriptors = np.asarray(saved["descriptors"], dtype=np.float32)
        filename = str(saved["filename"].item()) if "filename" in saved else None
        keypoint_info = (
            np.asarray(saved["keypoint_info"], dtype=np.float32)
            if "keypoint_info" in saved else None
        )

    if descriptors.ndim != 2:
        raise ValueError(f"Expected a 2-D descriptor matrix in {path}; got {descriptors.shape}.")
    if expected_dimension is not None and descriptors.shape[1] != expected_dimension:
        raise ValueError(
            f"Expected {expected_dimension} descriptor columns in {path}; "
            f"found {descriptors.shape[1]}."
        )
    if expected_filename is not None and filename is not None and filename != expected_filename:
        raise ValueError(f"Cache filename mismatch in {path}: {filename}.")
    if not np.isfinite(descriptors).all():
        raise ValueError(f"Non-finite descriptor values found in {path}.")
    if include_keypoints:
        if keypoint_info is None:
            raise KeyError(f"No keypoint_info array was saved in {path}.")
        return descriptors, keypoint_info
    return descriptors


def cache_gallery_features(
    records: Sequence[ImageRecord],
    cache_folder: str | Path,
    extractor: Callable,
    expected_dimension: int,
    rebuild: bool = False,
    progress_every: int = 50,
) -> list[int]:
    """Extract and cache one local-feature representation for every image.

    The extractor must accept an RGB image and return (keypoint_info, descriptors).
    Existing valid cache files are reused unless `rebuild=True`.
    """
    cache_folder = Path(cache_folder)
    cache_folder.mkdir(parents=True, exist_ok=True)
    counts = []
    errors = []

    for index, record in enumerate(records):
        path = feature_cache_path(cache_folder, index, record)
        try:
            if path.exists() and not rebuild:
                # SIFT caches also need keypoint locations for the later
                # OpponentSIFT stage, so validate both arrays when reusing them.
                descriptors, _ = load_feature_cache(
                    path, expected_filename=record.filename,
                    expected_dimension=expected_dimension, include_keypoints=True,
                )
            else:
                image_rgb = read_rgb(record.path)
                keypoint_info, descriptors = extractor(image_rgb)
                descriptors = np.asarray(descriptors, dtype=np.float32)
                if descriptors.ndim != 2 or descriptors.shape[1] != expected_dimension:
                    raise ValueError(f"Unexpected descriptor shape: {descriptors.shape}")
                save_feature_cache(path, record, descriptors, keypoint_info)

            counts.append(int(descriptors.shape[0]))
        except Exception as exc:
            errors.append((record.filename, str(exc)))

        if (index + 1) % progress_every == 0 or index + 1 == len(records):
            print(f"Processed {index + 1}/{len(records)} images")

    if errors:
        examples = "\n".join(f"{name}: {message}" for name, message in errors[:5])
        raise RuntimeError(f"Feature extraction failed for {len(errors)} images:\n{examples}")
    return counts


def cache_opponent_features(
    records: Sequence[ImageRecord],
    source_cache_folder: str | Path,
    output_cache_folder: str | Path,
    rebuild: bool = False,
    progress_every: int = 50,
) -> list[int]:
    """Create and cache 384-D OpponentSIFT using existing SIFT keypoint locations."""
    source_cache_folder = Path(source_cache_folder)
    output_cache_folder = Path(output_cache_folder)
    output_cache_folder.mkdir(parents=True, exist_ok=True)
    sift = create_sift()
    counts = []
    errors = []

    for index, record in enumerate(records):
        source_path = feature_cache_path(source_cache_folder, index, record)
        output_path = feature_cache_path(output_cache_folder, index, record)
        try:
            if output_path.exists() and not rebuild:
                descriptors = load_feature_cache(
                    output_path, expected_filename=record.filename,
                    expected_dimension=384,
                )
            else:
                _, keypoint_info = load_feature_cache(
                    source_path, expected_filename=record.filename,
                    expected_dimension=128, include_keypoints=True,
                )
                image_rgb = read_rgb(record.path)
                descriptors = extract_opponent_sift(image_rgb, keypoint_info, sift=sift)
                save_feature_cache(output_path, record, descriptors)
            counts.append(int(descriptors.shape[0]))
        except Exception as exc:
            errors.append((record.filename, str(exc)))

        if (index + 1) % progress_every == 0 or index + 1 == len(records):
            print(f"Processed {index + 1}/{len(records)} images")

    if errors:
        examples = "\n".join(f"{name}: {message}" for name, message in errors[:5])
        raise RuntimeError(f"OpponentSIFT extraction failed for {len(errors)} images:\n{examples}")
    return counts


def validate_cache_set(
    records: Sequence[ImageRecord],
    cache_folder: str | Path,
    expected_dimension: int,
    require_keypoints: bool = False,
) -> dict:
    """Check cache coverage, filenames and descriptor shapes for a split."""
    missing, invalid = [], []
    counts = []
    for index, record in enumerate(records):
        path = feature_cache_path(cache_folder, index, record)
        if not path.exists():
            missing.append(record.filename)
            continue
        try:
            descriptors = load_feature_cache(
                path, expected_filename=record.filename,
                expected_dimension=expected_dimension,
                include_keypoints=require_keypoints,
            )
            if require_keypoints:
                descriptors = descriptors[0]
            counts.append(int(descriptors.shape[0]))
        except Exception as exc:
            invalid.append((record.filename, str(exc)))

    if missing or invalid:
        raise RuntimeError(
            f"Cache check failed: {len(missing)} missing, {len(invalid)} invalid. "
            f"Missing examples: {missing[:3]}; invalid examples: {invalid[:3]}"
        )
    return {
        "images": len(records),
        "descriptor_dimension": expected_dimension,
        "minimum_descriptors": min(counts) if counts else 0,
        "mean_descriptors": float(np.mean(counts)) if counts else 0.0,
        "maximum_descriptors": max(counts) if counts else 0,
    }
