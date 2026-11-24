"""Reusable code for the EEE3032 MSRC-v2 visual-search investigation."""

from .features import (
    create_sift,
    extract_detected_sift,
    extract_dense_sift,
    make_dense_keypoints,
    root_sift,
    opponent_channels,
    extract_opponent_sift,
)
from .cache import (
    feature_cache_path,
    save_feature_cache,
    load_feature_cache,
    cache_gallery_features,
    cache_opponent_features,
)
from .vocabulary import (
    VOCABULARY_CONFIGS,
    build_training_matrix,
    train_vocabulary,
    train_all_vocabularies,
    vocabulary_path,
)

__all__ = [
    "create_sift", "extract_detected_sift", "extract_dense_sift",
    "make_dense_keypoints", "root_sift", "opponent_channels",
    "extract_opponent_sift", "feature_cache_path", "save_feature_cache",
    "load_feature_cache", "cache_gallery_features",
    "cache_opponent_features", "VOCABULARY_CONFIGS",
    "build_training_matrix", "train_vocabulary",
    "train_all_vocabularies", "vocabulary_path",
]
