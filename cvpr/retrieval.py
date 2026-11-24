from __future__ import annotations

from typing import Callable, Iterable, List

import numpy as np

from .dataset import ImageRecord, read_rgb


DescriptorFn = Callable[[np.ndarray], np.ndarray]


def compute_descriptors(records: Iterable[ImageRecord], descriptor_fn: DescriptorFn) -> np.ndarray:
    """Compute one fixed-length descriptor per image record."""
    features: List[np.ndarray] = []
    expected_dim = None

    for i, record in enumerate(records, start=1):
        feature = np.asarray(descriptor_fn(read_rgb(record.path)), dtype=np.float32).ravel()
        if expected_dim is None:
            expected_dim = feature.size
        elif feature.size != expected_dim:
            raise ValueError(
                f"Descriptor length changed from {expected_dim} to {feature.size} for {record.filename}."
            )
        features.append(feature)

    if not features:
        raise ValueError("No records were supplied.")
    return np.vstack(features)


def euclidean_distances(query_feature: np.ndarray, gallery_features: np.ndarray) -> np.ndarray:
    """Euclidean distance from one query descriptor to every gallery descriptor."""
    q = np.asarray(query_feature, dtype=np.float32).ravel()
    g = np.asarray(gallery_features, dtype=np.float32)
    if g.ndim != 2 or g.shape[1] != q.size:
        raise ValueError("Query and gallery descriptor dimensions do not match.")
    return np.linalg.norm(g - q[None, :], axis=1)


def rank_gallery(query_feature: np.ndarray, gallery_features: np.ndarray) -> np.ndarray:
    """Return gallery indices sorted from most to least similar for the baseline metric."""
    return np.argsort(euclidean_distances(query_feature, gallery_features))
