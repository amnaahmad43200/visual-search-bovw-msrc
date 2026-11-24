from __future__ import annotations

import math
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from .dataset import ImageRecord, read_rgb
from .retrieval import euclidean_distances


def show_ranked_results(
    query_record: ImageRecord,
    query_feature: np.ndarray,
    gallery_records: Sequence[ImageRecord],
    gallery_features: np.ndarray,
    top_k: int = 10,
    distance_fn=euclidean_distances,
):
    distances = np.asarray(distance_fn(query_feature, gallery_features), dtype=float)
    order = np.argsort(distances)[:top_k]

    ncols = 5
    nrows = 1 + math.ceil(top_k / ncols)
    fig = plt.figure(figsize=(14, 3 * nrows))

    ax = plt.subplot(nrows, ncols, 1)
    ax.imshow(read_rgb(query_record.path))
    ax.set_title(f"QUERY\n{query_record.filename}\nGroup {query_record.category}")
    ax.axis("off")

    for rank, gi in enumerate(order, start=1):
        ax = plt.subplot(nrows, ncols, ncols + rank)
        rec = gallery_records[gi]
        ax.imshow(read_rgb(rec.path))
        ok = "relevant" if rec.category == query_record.category else "non-relevant"
        ax.set_title(f"#{rank} {rec.filename}\n{ok}, d={distances[gi]:.3f}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    return fig, order



def plot_confusion_matrix(matrix, categories, *, normalize=False, title="Top-1 category confusion matrix"):
    """Display a category confusion matrix. Rows=true query group; columns=top-1 retrieved group."""
    m = np.asarray(matrix)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError("matrix must be square.")
    shown = m.astype(float)
    if normalize:
        denom = shown.sum(axis=1, keepdims=True)
        shown = np.divide(shown, denom, out=np.zeros_like(shown), where=denom != 0)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(shown, interpolation="nearest", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Category of top-ranked retrieval")
    ax.set_ylabel("Query category")
    ax.set_xticks(range(len(categories)))
    ax.set_yticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=90)
    ax.set_yticklabels(categories)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Fraction of queries" if normalize else "Number of queries")
    fig.tight_layout()
    return fig, ax
