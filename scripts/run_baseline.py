from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]  # repository root
sys.path.insert(0, str(ROOT))

from cvpr.dataset import load_split_records
from cvpr.baseline import global_rgb_histogram
from cvpr.retrieval import compute_descriptors
from cvpr.evaluation import evaluate_retrieval, per_category_summary


def main():
    parser = argparse.ArgumentParser(description="Run the supplied EEE3032 visual-search baseline.")
    parser.add_argument("dataset", help="Extracted MSRC dataset folder or supplied dataset ZIP")
    parser.add_argument("--bins", type=int, default=4, help="Histogram bins per RGB channel (default: 4)")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    args = parser.parse_args()

    root = ROOT
    manifest = root / "split_manifest.csv"
    gallery, dev, test = load_split_records(args.dataset, manifest_path=manifest, extract_dir=root / "data")
    queries = dev if args.split == "dev" else test

    descriptor = partial(global_rgb_histogram, bins_per_channel=args.bins)
    gallery_features = compute_descriptors(gallery, descriptor)
    query_features = compute_descriptors(queries, descriptor)

    summary, per_query = evaluate_retrieval(queries, query_features, gallery, gallery_features, k=10)
    print(f"Split: {args.split}")
    print(f"Descriptor dimension: {gallery_features.shape[1]}")
    print(f"mAP: {summary['mAP']:.4f}")
    print(f"mean P@10: {summary['mean_P@10']:.4f}")

    print("\nPer-category results:")
    for row in per_category_summary(per_query):
        print(
            f"Group {row['category']:>2}: mAP={row['mAP']:.4f}, "
            f"mean P@10={row['mean_P@10']:.4f}"
        )


if __name__ == "__main__":
    main()
