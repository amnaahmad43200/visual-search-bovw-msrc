from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import os
import re
import zipfile
from typing import Iterable, List, Tuple

import cv2


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    filename: str
    category: int
    split: str


def _find_images_dir(root: Path) -> Path | None:
    """Find the MSRC Images folder below *root* without assuming one exact layout."""
    root = root.expanduser().resolve()
    if root.is_dir() and root.name.lower() == "images":
        return root
    if (root / "Images").is_dir():
        return root / "Images"
    if root.is_dir():
        candidates = [p for p in root.rglob("Images") if p.is_dir()]
        for p in candidates:
            if any(p.glob("*.bmp")):
                return p
    return None


def prepare_dataset(dataset_source: str | Path, extract_dir: str | Path = "data") -> Path:
    """
    Return the path to the MSRC Images directory.

    dataset_source may be:
      * the extracted dataset root,
      * the Images directory itself, or
      * the supplied MSRC .zip file.
    """
    src = Path(dataset_source).expanduser()

    if src.exists() and src.is_dir():
        found = _find_images_dir(src)
        if found is None:
            raise FileNotFoundError(f"Could not find an Images folder below: {src}")
        return found

    if src.exists() and src.is_file() and src.suffix.lower() == ".zip":
        extract_dir = Path(extract_dir).expanduser().resolve()
        extract_dir.mkdir(parents=True, exist_ok=True)

        # First see whether this ZIP directly contains the dataset.
        with zipfile.ZipFile(src, "r") as zf:
            names = zf.namelist()
            direct = any("/Images/" in name and name.lower().endswith(".bmp") for name in names)
            if direct:
                zf.extractall(extract_dir)
            else:
                # Some downloaded copies wrap the actual dataset ZIP inside another ZIP.
                nested = [n for n in names if n.lower().endswith(".zip")]
                if not nested:
                    raise FileNotFoundError("The ZIP does not appear to contain the MSRC dataset.")
                nested_name = nested[0]
                nested_path = extract_dir / Path(nested_name).name
                if not nested_path.exists():
                    nested_path.write_bytes(zf.read(nested_name))
                with zipfile.ZipFile(nested_path, "r") as inner:
                    inner.extractall(extract_dir)

        found = _find_images_dir(extract_dir)
        if found is None:
            raise FileNotFoundError("Dataset extracted, but no Images folder was found.")
        return found

    raise FileNotFoundError(
        f"Dataset source not found: {src}\n"
        "Set DATASET_SOURCE to either the extracted MSRC folder or the supplied dataset ZIP."
    )


def read_rgb(path: str | Path):
    """Read an image as uint8 RGB."""
    path = Path(path)
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_split_records(
    dataset_source: str | Path,
    manifest_path: str | Path = "split_manifest.csv",
    extract_dir: str | Path = "data",
) -> Tuple[List[ImageRecord], List[ImageRecord], List[ImageRecord]]:
    """Load the official gallery/development/test split supplied with the coursework."""
    images_dir = prepare_dataset(dataset_source, extract_dir=extract_dir)
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")

    rows = []
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    records: List[ImageRecord] = []
    missing = []
    for row in rows:
        filename = row["filename"]
        path = images_dir / filename
        if not path.exists():
            missing.append(filename)
            continue
        records.append(
            ImageRecord(
                path=path,
                filename=filename,
                category=int(row["category"]),
                split=row["split"],
            )
        )

    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} dataset images listed in the manifest were not found. "
            f"First missing files: {preview}"
        )

    gallery = [r for r in records if r.split == "gallery"]
    dev = [r for r in records if r.split == "dev"]
    test = [r for r in records if r.split == "test"]

    expected = (391, 100, 100)
    if (len(gallery), len(dev), len(test)) != expected:
        raise ValueError(
            "Unexpected split sizes: "
            f"gallery={len(gallery)}, dev={len(dev)}, test={len(test)}; expected {expected}."
        )

    return gallery, dev, test
