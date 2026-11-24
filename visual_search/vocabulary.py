"""Gallery-only visual vocabulary training for the SIFT/BoVW experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Sequence
import csv

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from cvpr.dataset import ImageRecord

from .cache import feature_cache_path, load_feature_cache
from .features import root_sift


@dataclass(frozen=True)
class VocabularyConfig:
    """Identify one sampling and descriptor representation."""
    name: str
    sampling: str
    source: str
    use_root: bool
    dimension: int


# Each representation needs its own vocabulary because descriptor dimensions
# and values differ between the configurations.
VOCABULARY_CONFIGS = (
    VocabularyConfig("detected_sift", "detected", "sift", False, 128),
    VocabularyConfig("detected_rootsift", "detected", "sift", True, 128),
    VocabularyConfig("detected_opponentsift", "detected", "opponent", False, 384),
    VocabularyConfig("detected_root_opponentsift", "detected", "opponent", True, 384),
    VocabularyConfig("dense_sift", "dense", "sift", False, 128),
    VocabularyConfig("dense_rootsift", "dense", "sift", True, 128),
    VocabularyConfig("dense_opponentsift", "dense", "opponent", False, 384),
    VocabularyConfig("dense_root_opponentsift", "dense", "opponent", True, 384),
)


def descriptor_cache_folder(project_root: str | Path, config: VocabularyConfig) -> Path:
    """Return the cache directory created by the feature-extraction stages."""
    root = Path(project_root)
    if config.source == "sift":
        if config.sampling == "detected":
            return root / "sift_cache" / "detected_sift"
        return root / "sift_cache" / "dense_sift_step12_size16"

    if config.sampling == "detected":
        return root / "sift_cache" / "stage3_opponent_sift" / "detected"
    return root / "sift_cache" / "stage3_opponent_sift" / "dense"


def vocabulary_path(
    project_root: str | Path,
    config: VocabularyConfig,
    k: int = 100,
    descriptor_cap: int = 200,
    random_seed: int = 3032,
) -> Path:
    """Return the standard filename for a trained vocabulary."""
    filename = (
        f"vocab_{config.name}_K{k}_cap{descriptor_cap}_seed{random_seed}.npz"
    )
    return Path(project_root) / "bovw_vocabularies" / filename


def load_image_descriptors(
    project_root: str | Path,
    image_index: int,
    record: ImageRecord,
    config: VocabularyConfig,
) -> np.ndarray:
    """Load one cached descriptor matrix and apply RootSIFT when requested."""
    cache_file = feature_cache_path(
        descriptor_cache_folder(project_root, config), image_index, record
    )
    descriptors = load_feature_cache(
        cache_file,
        expected_filename=record.filename,
        expected_dimension=config.dimension,
    )
    if config.use_root:
        # For RootOpponentSIFT this transforms the concatenated 384 values
        # as one descriptor, matching the representation used in Stage 4.
        descriptors = root_sift(descriptors)
    return descriptors


def sample_descriptors(
    descriptors: np.ndarray,
    image_index: int,
    maximum: int = 200,
    random_seed: int = 3032,
) -> np.ndarray:
    """Select at most `maximum` rows from one image, reproducibly."""
    values = np.asarray(descriptors, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("Descriptors must be a 2-D array.")
    if maximum <= 0:
        raise ValueError("maximum must be positive.")
    if values.shape[0] <= maximum:
        return values

    # A per-image seed ensures that corresponding SIFT/RootSIFT variants
    # select the same descriptor rows when their row counts are identical.
    rng = np.random.default_rng(random_seed + image_index)
    chosen = rng.choice(values.shape[0], size=maximum, replace=False)
    chosen.sort()
    return values[chosen]


def build_training_matrix(
    gallery_records: Sequence[ImageRecord],
    project_root: str | Path,
    config: VocabularyConfig,
    maximum_per_image: int = 200,
    random_seed: int = 3032,
) -> tuple[np.ndarray, dict]:
    """Build the K-means input using gallery descriptors only."""
    if not gallery_records:
        raise ValueError("The gallery record list is empty.")
    if any(record.split != "gallery" for record in gallery_records):
        raise ValueError("Vocabulary fitting must use gallery records only.")

    selected_rows = []
    selected_counts = []
    empty_images = []

    for image_index, record in enumerate(gallery_records):
        descriptors = load_image_descriptors(project_root, image_index, record, config)
        if descriptors.shape[0] == 0:
            empty_images.append(record.filename)
            selected_counts.append(0)
            continue

        sampled = sample_descriptors(
            descriptors, image_index, maximum_per_image, random_seed
        )
        selected_rows.append(sampled)
        selected_counts.append(int(sampled.shape[0]))

    if not selected_rows:
        raise ValueError(f"No descriptors were available for {config.name}.")

    training_data = np.concatenate(selected_rows, axis=0).astype(np.float32, copy=False)
    if not np.isfinite(training_data).all():
        raise ValueError("The K-means training matrix contains non-finite values.")

    # Shuffle the matrix so batches are not ordered by source image.
    shuffle_rng = np.random.default_rng(random_seed + 9999)
    shuffle_rng.shuffle(training_data, axis=0)

    details = {
        "training_descriptors": int(training_data.shape[0]),
        "descriptor_dimension": int(training_data.shape[1]),
        "minimum_per_image": int(min(selected_counts)),
        "maximum_per_image": int(max(selected_counts)),
        "images_with_no_descriptors": len(empty_images),
        "empty_image_names": empty_images,
    }
    return training_data, details


def train_vocabulary(
    gallery_records: Sequence[ImageRecord],
    project_root: str | Path,
    config: VocabularyConfig,
    k: int = 100,
    maximum_per_image: int = 200,
    random_seed: int = 3032,
    batch_size: int = 2048,
    max_iter: int = 100,
    n_init: int = 3,
    rebuild: bool = False,
) -> dict:
    """Fit and save one MiniBatchKMeans visual vocabulary."""
    if k <= 0:
        raise ValueError("k must be positive.")

    output_file = vocabulary_path(project_root, config, k, maximum_per_image, random_seed)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Reuse a vocabulary only when its core settings match this run.
    if output_file.exists() and not rebuild:
        with np.load(output_file, allow_pickle=False) as saved:
            centres = np.asarray(saved["centres"], dtype=np.float32)
            matching = (
                int(saved["K"]) == k
                and int(saved["descriptor_cap"]) == maximum_per_image
                and int(saved["random_seed"]) == random_seed
                and str(saved["configuration"].item()) == config.name
                and int(saved["batch_size"]) == min(batch_size, int(saved["training_descriptors"]))
                and int(saved["max_iter"]) == max_iter
                and int(saved["n_init"]) == n_init
                and centres.shape == (k, config.dimension)
            )
            if matching:
                cluster_sizes = np.asarray(saved["cluster_sizes"], dtype=np.int64)
                return {
                    "configuration": config.name,
                    "status": "reused",
                    "training_descriptors": int(saved["training_descriptors"]),
                    "descriptor_dimension": config.dimension,
                    "vocabulary_shape": str(centres.shape),
                    "fit_seconds": float(saved["fit_seconds"]),
                    "iterations": int(saved["iterations"]),
                    "inertia": float(saved["inertia"]),
                    "empty_clusters": int(np.sum(cluster_sizes == 0)),
                    "vocabulary_file": str(output_file),
                }

    training_data, details = build_training_matrix(
        gallery_records, project_root, config, maximum_per_image, random_seed
    )
    if training_data.shape[0] < k:
        raise ValueError(
            f"Only {training_data.shape[0]} training descriptors are available for K={k}."
        )

    effective_batch_size = min(batch_size, training_data.shape[0])
    model = MiniBatchKMeans(
        n_clusters=k,
        init="k-means++",
        batch_size=effective_batch_size,
        max_iter=max_iter,
        n_init=n_init,
        random_state=random_seed,
        reassignment_ratio=0.01,
        max_no_improvement=10,
    )

    start = perf_counter()
    model.fit(training_data)
    fit_seconds = perf_counter() - start

    centres = model.cluster_centers_.astype(np.float32)
    labels = model.predict(training_data)
    cluster_sizes = np.bincount(labels, minlength=k).astype(np.int64)

    if centres.shape != (k, config.dimension) or not np.isfinite(centres).all():
        raise RuntimeError(f"Invalid vocabulary centres for {config.name}.")

    np.savez_compressed(
        output_file,
        centres=centres,
        configuration=np.asarray(config.name),
        sampling=np.asarray(config.sampling),
        representation=np.asarray(config.source),
        root_transform=np.asarray(config.use_root),
        K=np.asarray(k),
        descriptor_cap=np.asarray(maximum_per_image),
        random_seed=np.asarray(random_seed),
        batch_size=np.asarray(effective_batch_size),
        max_iter=np.asarray(max_iter),
        n_init=np.asarray(n_init),
        training_descriptors=np.asarray(training_data.shape[0]),
        fit_seconds=np.asarray(fit_seconds),
        inertia=np.asarray(float(model.inertia_)),
        iterations=np.asarray(int(model.n_iter_)),
        cluster_sizes=cluster_sizes,
    )

    result = {
        "configuration": config.name,
        "status": "trained",
        "training_descriptors": int(training_data.shape[0]),
        "descriptor_dimension": config.dimension,
        "vocabulary_shape": str(centres.shape),
        "fit_seconds": round(float(fit_seconds), 2),
        "iterations": int(model.n_iter_),
        "inertia": float(model.inertia_),
        "empty_clusters": int(np.sum(cluster_sizes == 0)),
        "vocabulary_file": str(output_file),
    }
    return result


def train_all_vocabularies(
    gallery_records: Sequence[ImageRecord],
    project_root: str | Path,
    configs: Sequence[VocabularyConfig] = VOCABULARY_CONFIGS,
    **training_options,
) -> list[dict]:
    """Train or reuse each configured vocabulary, reporting progress."""
    results = []
    for index, config in enumerate(configs, start=1):
        print(f"[{index}/{len(configs)}] {config.name}")
        result = train_vocabulary(
            gallery_records, project_root, config, **training_options
        )
        results.append(result)
        print(
            f"  {result['status']}: {result['vocabulary_shape']}, "
            f"{result['training_descriptors']} training descriptors, "
            f"{result['fit_seconds']:.2f}s, "
            f"{result['empty_clusters']} empty clusters"
        )
    return results


def save_vocabulary_summary(results: Sequence[dict], output_file: str | Path) -> Path:
    """Write a compact CSV summary of the trained vocabularies."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if not results:
        raise ValueError("There are no vocabulary results to save.")
    fieldnames = list(results[0].keys())
    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    return output_file
