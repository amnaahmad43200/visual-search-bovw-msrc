"""Helper code supplied for the EEE3032 visual-search coursework."""

from .dataset import ImageRecord, load_split_records, read_rgb
from .baseline import global_rgb_histogram
from .retrieval import compute_descriptors, euclidean_distances, rank_gallery
from .evaluation import evaluate_retrieval, per_category_summary

from .personal import load_personal_extension
from .deep import (
    DeepDescriptorExtractor,
    deep_dependencies_available,
    make_mobilenet_v3_small_extractor,
    make_squeezenet1_1_extractor,
)
from .evaluation import top1_confusion_matrix
from .plotting import plot_confusion_matrix
