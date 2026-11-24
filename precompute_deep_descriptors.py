from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from cvpr.dataset import load_split_records
from cvpr.retrieval import compute_descriptors
from cvpr.deep import (
    make_mobilenet_v3_small_extractor,
    make_squeezenet1_1_extractor,
    save_feature_cache,
)


def main():
    parser = argparse.ArgumentParser(description="Precompute optional EEE3032 deep descriptors for SurreyLearn distribution.")
    parser.add_argument("dataset_source", help="Extracted MSRC folder or dataset ZIP")
    parser.add_argument("--output", default="deep_cache", help="Output directory")
    parser.add_argument("--model", choices=["mobilenet_v3_small", "squeezenet1_1", "both"], default="both")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    gallery, dev, test = load_split_records(args.dataset_source, root / "split_manifest.csv", root / "data")
    all_records = list(gallery) + list(dev) + list(test)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    names = ["mobilenet_v3_small", "squeezenet1_1"] if args.model == "both" else [args.model]
    for name in names:
        print("Computing", name)
        extractor = (
            make_mobilenet_v3_small_extractor()
            if name == "mobilenet_v3_small"
            else make_squeezenet1_1_extractor()
        )
        features = compute_descriptors(all_records, extractor)
        save_feature_cache(out / f"{name}_all_msrc.npz", all_records, features, descriptor_name=name)
        print("Saved", out / f"{name}_all_msrc.npz", features.shape)


if __name__ == "__main__":
    main()
