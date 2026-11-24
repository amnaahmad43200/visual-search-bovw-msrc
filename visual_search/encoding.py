"""Bag-of-Visual-Words encoding and development-set evaluation helpers.

The functions in this module convert local descriptors into fixed-length
visual-word histograms. Vocabulary and IDF fitting are kept separate so that
both are fitted on gallery images only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from cvpr.dataset import ImageRecord

from .cache import (
    cache_gallery_features,
    feature_cache_path,
    load_feature_cache,
)
from .features import (
    create_sift,
    extract_detected_sift,
    extract_dense_sift,
    extract_opponent_sift,
    root_sift,
)
from .vocabulary import VocabularyConfig


def query_cache_folder(
    project_root: str | Path,
    split_name: str,
    sampling: str,
    representation: str,
) -> Path:
    """Return a separate cache directory for one query split and feature type.

    Query descriptors are cached separately from gallery descriptors. This
    prevents their per-split image indices from colliding with gallery indices.
    """
    if split_name not in {"dev", "test"}:
        raise ValueError("split_name must be 'dev' or 'test'.")
    if sampling not in {"detected", "dense"}:
        raise ValueError("sampling must be 'detected' or 'dense'.")
    if representation not in {"sift", "opponent"}:
        raise ValueError("representation must be 'sift' or 'opponent'.")

    return (
        Path(project_root)
        / "sift_cache"
        / "stage5_queries"
        / split_name
        / f"{sampling}_{representation}"
    )


def extract_base_descriptors(
    image_rgb: np.ndarray,
    sampling: str,
    representation: str,
    sift=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract raw SIFT or OpponentSIFT descriptors at one set of locations.

    Root normalisation is deliberately not applied here. Keeping raw query
    descriptors in the cache means standard and Root variants can reuse the
    same extraction result.
    """
    if sampling == "detected":
        keypoint_info, sift_descriptors = extract_detected_sift(image_rgb, sift=sift)
    elif sampling == "dense":
        keypoint_info, sift_descriptors = extract_dense_sift(image_rgb, sift=sift)
    else:
        raise ValueError("sampling must be 'detected' or 'dense'.")

    if representation == "sift":
        descriptors = sift_descriptors
    elif representation == "opponent":
        descriptors = extract_opponent_sift(
            image_rgb, keypoint_info, sift=sift
        )
    else:
        raise ValueError("representation must be 'sift' or 'opponent'.")

    return keypoint_info, np.asarray(descriptors, dtype=np.float32)


def cache_query_features(
    records: Sequence[ImageRecord],
    project_root: str | Path,
    split_name: str,
    sampling: str,
    representation: str,
    rebuild: bool = False,
    progress_every: int = 25,
) -> list[int]:
    """Extract and cache one raw descriptor representation for a query split.

    This is used for development queries in Stage 5. The same helper can cache
    test descriptors later, after the final method has been selected.
    """
    if not records:
        raise ValueError("No image records were supplied.")
    if any(record.split != split_name for record in records):
        raise ValueError(f"All records must belong to the '{split_name}' split.")

    cache_folder = query_cache_folder(
        project_root, split_name, sampling, representation
    )
    sift = create_sift()

    def extractor(image_rgb: np.ndarray):
        return extract_base_descriptors(
            image_rgb, sampling=sampling,
            representation=representation, sift=sift,
        )

    expected_dimension = 128 if representation == "sift" else 384
    return cache_gallery_features(
        records,
        cache_folder,
        extractor=extractor,
        expected_dimension=expected_dimension,
        rebuild=rebuild,
        progress_every=progress_every,
    )


def load_vocabulary_centres(
    vocabulary_file: str | Path,
    expected_configuration: str,
    expected_dimension: int,
    expected_k: int,
    expected_descriptor_cap: int | None = None,
    expected_random_seed: int | None = None,
) -> np.ndarray:
    """Load and validate the cluster centres saved during Stage 4.

    Optional checks on the descriptor cap and random seed help catch a stale
    vocabulary that happens to have the right filename and matrix dimensions.
    """
    vocabulary_file = Path(vocabulary_file)
    if not vocabulary_file.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {vocabulary_file}")

    with np.load(vocabulary_file, allow_pickle=False) as saved:
        centres = np.asarray(saved["centres"], dtype=np.float32)
        configuration = str(saved["configuration"].item())
        saved_k = int(saved["K"])
        saved_cap = int(saved["descriptor_cap"])
        saved_seed = int(saved["random_seed"])

    if configuration != expected_configuration:
        raise ValueError(
            f"Expected vocabulary '{expected_configuration}', found '{configuration}'."
        )
    if saved_k != expected_k or centres.shape != (expected_k, expected_dimension):
        raise ValueError(
            f"Unexpected vocabulary shape {centres.shape}; expected "
            f"({expected_k}, {expected_dimension})."
        )
    if expected_descriptor_cap is not None and saved_cap != expected_descriptor_cap:
        raise ValueError(
            f"Vocabulary descriptor cap is {saved_cap}; expected {expected_descriptor_cap}."
        )
    if expected_random_seed is not None and saved_seed != expected_random_seed:
        raise ValueError(
            f"Vocabulary random seed is {saved_seed}; expected {expected_random_seed}."
        )
    if not np.isfinite(centres).all():
        raise ValueError("Vocabulary contains NaN or infinite values.")

    return centres


def load_descriptor_matrices(
    records: Sequence[ImageRecord],
    cache_folder: str | Path,
    expected_dimension: int,
    use_root: bool = False,
) -> list[np.ndarray]:
    """Load cached descriptors in record order and optionally apply RootSIFT."""
    matrices = []
    for image_index, record in enumerate(records):
        cache_file = feature_cache_path(cache_folder, image_index, record)
        descriptors = load_feature_cache(
            cache_file,
            expected_filename=record.filename,
            expected_dimension=expected_dimension,
        )
        if use_root:
            descriptors = root_sift(descriptors)
        matrices.append(np.asarray(descriptors, dtype=np.float32))
    return matrices


def assign_visual_words(
    descriptors: np.ndarray,
    centres: np.ndarray,
    batch_size: int = 2048,
) -> np.ndarray:
    """Assign each local descriptor to its nearest visual-word centre.

    Squared Euclidean distances are used. Taking the square root is not
    necessary because it cannot change which centre is nearest. Processing
    descriptors in batches avoids creating an unnecessarily large distance
    matrix for images with many keypoints.
    """
    values = np.asarray(descriptors, dtype=np.float32)
    vocabulary = np.asarray(centres, dtype=np.float32)

    if values.ndim != 2 or vocabulary.ndim != 2:
        raise ValueError("Descriptors and centres must both be 2-D arrays.")
    if values.shape[1] != vocabulary.shape[1]:
        raise ValueError("Descriptor and vocabulary dimensions do not match.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if not np.isfinite(values).all() or not np.isfinite(vocabulary).all():
        raise ValueError("Descriptors and vocabulary must contain finite values.")
    if values.shape[0] == 0:
        return np.empty(0, dtype=np.int32)

    centre_squared_norms = np.sum(vocabulary * vocabulary, axis=1)
    assignments = np.empty(values.shape[0], dtype=np.int32)

    for start in range(0, values.shape[0], batch_size):
        stop = min(start + batch_size, values.shape[0])
        batch = values[start:stop]
        descriptor_squared_norms = np.sum(batch * batch, axis=1, keepdims=True)

        # ||x-c||^2 = ||x||^2 + ||c||^2 - 2(x dot c)
        distances = (
            descriptor_squared_norms
            + centre_squared_norms[None, :]
            - 2.0 * (batch @ vocabulary.T)
        )
        np.maximum(distances, 0.0, out=distances)
        assignments[start:stop] = np.argmin(distances, axis=1)

    return assignments


def build_bovw_histogram(
    descriptors: np.ndarray,
    centres: np.ndarray,
    assignment_batch_size: int = 2048,
) -> np.ndarray:
    """Count the visual words assigned to one image's local descriptors."""
    vocabulary = np.asarray(centres, dtype=np.float32)
    if vocabulary.ndim != 2:
        raise ValueError("Vocabulary centres must be a 2-D array.")

    assignments = assign_visual_words(
        descriptors, vocabulary, batch_size=assignment_batch_size
    )
    return np.bincount(
        assignments, minlength=vocabulary.shape[0]
    ).astype(np.float32)


def encode_cached_records(
    records: Sequence[ImageRecord],
    cache_folder: str | Path,
    config: VocabularyConfig,
    centres: np.ndarray,
    assignment_batch_size: int = 2048,
) -> np.ndarray:
    """Convert a split's cached local descriptors into BoVW count histograms.

    Every descriptor in each image is used here. The 200-descriptor cap from
    Stage 4 applies only to K-means vocabulary training, not image encoding.
    """
    vocabulary = np.asarray(centres, dtype=np.float32)
    if vocabulary.shape[1] != config.dimension:
        raise ValueError("Vocabulary dimension does not match the configuration.")

    histograms = np.zeros((len(records), vocabulary.shape[0]), dtype=np.float32)
    for image_index, record in enumerate(records):
        cache_file = feature_cache_path(cache_folder, image_index, record)
        descriptors = load_feature_cache(
            cache_file,
            expected_filename=record.filename,
            expected_dimension=config.dimension,
        )

        if config.use_root:
            descriptors = root_sift(descriptors)

        histograms[image_index] = build_bovw_histogram(
            descriptors,
            vocabulary,
            assignment_batch_size=assignment_batch_size,
        )

    return histograms


def fit_idf(gallery_histograms: np.ndarray) -> np.ndarray:
    """Fit inverse-document-frequency weights using gallery histograms only."""
    counts = np.asarray(gallery_histograms, dtype=np.float32)
    if counts.ndim != 2 or counts.shape[0] == 0:
        raise ValueError("Gallery histograms must be a non-empty 2-D matrix.")
    if np.any(counts < 0) or not np.isfinite(counts).all():
        raise ValueError("Gallery histograms must be finite and non-negative.")

    number_of_images = counts.shape[0]
    document_frequency = np.count_nonzero(counts > 0, axis=0)

    # Smooth the formula so a word that appears in no gallery image is valid.
    idf = np.log(
        (number_of_images + 1.0) / (document_frequency + 1.0)
    ) + 1.0
    return idf.astype(np.float32)


def apply_tfidf_l2(
    histograms: np.ndarray,
    idf: np.ndarray,
) -> np.ndarray:
    """Apply gallery-fitted IDF weights and L2-normalise each image vector."""
    counts = np.asarray(histograms, dtype=np.float32)
    weights = np.asarray(idf, dtype=np.float32).ravel()

    if counts.ndim != 2 or counts.shape[1] != weights.size:
        raise ValueError("Histogram and IDF dimensions do not match.")
    if np.any(counts < 0) or not np.isfinite(counts).all():
        raise ValueError("Histograms must be finite and non-negative.")
    if np.any(weights < 0) or not np.isfinite(weights).all():
        raise ValueError("IDF weights must be finite and non-negative.")

    weighted = counts * weights[None, :]
    row_norms = np.linalg.norm(weighted, axis=1, keepdims=True)
    return np.divide(
        weighted,
        row_norms,
        out=np.zeros_like(weighted),
        where=row_norms > 0,
    ).astype(np.float32)


def save_bovw_features(
    output_file: str | Path,
    configuration: str,
    vocabulary_file: str | Path,
    gallery_records: Sequence[ImageRecord],
    query_records: Sequence[ImageRecord],
    gallery_features: np.ndarray,
    query_features: np.ndarray,
    idf: np.ndarray,
    k: int,
) -> Path:
    """Save encoded gallery/query features and gallery-fitted IDF weights."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_file,
        configuration=np.asarray(configuration),
        vocabulary_file=np.asarray(str(vocabulary_file)),
        K=np.asarray(k),
        gallery_features=np.asarray(gallery_features, dtype=np.float32),
        query_features=np.asarray(query_features, dtype=np.float32),
        idf=np.asarray(idf, dtype=np.float32),
        gallery_filenames=np.asarray([r.filename for r in gallery_records]),
        query_filenames=np.asarray([r.filename for r in query_records]),
    )
    return output_file
