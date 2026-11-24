from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .dataset import ImageRecord
from .retrieval import euclidean_distances


DistanceFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def precision_at_k(relevance: np.ndarray, k: int = 10) -> float:
    relevance = np.asarray(relevance, dtype=bool)
    if k <= 0:
        raise ValueError("k must be positive.")
    k = min(k, relevance.size)
    return float(relevance[:k].mean()) if k else 0.0


def average_precision(relevance: np.ndarray, total_relevant: int | None = None) -> float:
    """Average precision for one ranked query result."""
    rel = np.asarray(relevance, dtype=bool)
    if total_relevant is None:
        total_relevant = int(rel.sum())
    if total_relevant <= 0:
        return 0.0

    hits = np.flatnonzero(rel)
    if hits.size == 0:
        return 0.0
    precisions = [(rank + 1) / (idx + 1) for rank, idx in enumerate(hits)]
    return float(np.sum(precisions) / total_relevant)


def evaluate_retrieval(
    query_records: Sequence[ImageRecord],
    query_features: np.ndarray,
    gallery_records: Sequence[ImageRecord],
    gallery_features: np.ndarray,
    distance_fn: DistanceFn = euclidean_distances,
    k: int = 10,
):
    """
    Evaluate a retrieval method using same-filename-group relevance.

    Returns (summary, per_query_rows).  The summary contains mAP and mean P@k.
    """
    qf = np.asarray(query_features)
    gf = np.asarray(gallery_features)
    if len(query_records) != len(qf):
        raise ValueError("Number of query records and query descriptors differs.")
    if len(gallery_records) != len(gf):
        raise ValueError("Number of gallery records and gallery descriptors differs.")

    gallery_categories = np.array([r.category for r in gallery_records], dtype=int)
    rows: List[Dict[str, float | int | str]] = []

    for qi, (record, feature) in enumerate(zip(query_records, qf)):
        distances = np.asarray(distance_fn(feature, gf), dtype=float)
        if distances.shape != (len(gallery_records),):
            raise ValueError("distance_fn must return one distance per gallery image.")
        order = np.argsort(distances)
        relevance = gallery_categories[order] == record.category
        n_relevant = int(np.sum(gallery_categories == record.category))

        rows.append(
            {
                "filename": record.filename,
                "category": record.category,
                "AP": average_precision(relevance, total_relevant=n_relevant),
                f"P@{k}": precision_at_k(relevance, k=k),
            }
        )

    aps = np.array([r["AP"] for r in rows], dtype=float)
    pks = np.array([r[f"P@{k}"] for r in rows], dtype=float)
    summary = {
        "n_queries": len(rows),
        "mAP": float(aps.mean()),
        f"mean_P@{k}": float(pks.mean()),
    }
    return summary, rows


def per_category_summary(per_query_rows, k: int = 10):
    grouped = defaultdict(list)
    for row in per_query_rows:
        grouped[int(row["category"])].append(row)

    result = []
    for category in sorted(grouped):
        rows = grouped[category]
        result.append(
            {
                "category": category,
                "n_queries": len(rows),
                "mAP": float(np.mean([r["AP"] for r in rows])),
                f"mean_P@{k}": float(np.mean([r[f"P@{k}"] for r in rows])),
            }
        )
    return result



def top1_confusion_matrix(
    query_records: Sequence[ImageRecord],
    query_features: np.ndarray,
    gallery_records: Sequence[ImageRecord],
    gallery_features: np.ndarray,
    distance_fn: DistanceFn = euclidean_distances,
    categories: Sequence[int] | None = None,
):
    """
    Category confusion matrix using the category of the top-ranked gallery image.

    Rows are true query categories; columns are the category of rank-1 retrieval.
    Returns (matrix, categories).
    """
    qf = np.asarray(query_features)
    gf = np.asarray(gallery_features)
    if len(query_records) != len(qf):
        raise ValueError("Number of query records and query descriptors differs.")
    if len(gallery_records) != len(gf):
        raise ValueError("Number of gallery records and gallery descriptors differs.")

    if categories is None:
        categories = sorted({int(r.category) for r in query_records} | {int(r.category) for r in gallery_records})
    categories = [int(c) for c in categories]
    index = {c: i for i, c in enumerate(categories)}
    matrix = np.zeros((len(categories), len(categories)), dtype=int)

    for record, feature in zip(query_records, qf):
        distances = np.asarray(distance_fn(feature, gf), dtype=float)
        if distances.shape != (len(gallery_records),):
            raise ValueError("distance_fn must return one distance per gallery image.")
        nearest = int(np.argmin(distances))
        true_cat = int(record.category)
        pred_cat = int(gallery_records[nearest].category)
        if true_cat in index and pred_cat in index:
            matrix[index[true_cat], index[pred_cat]] += 1

    return matrix, categories
