from __future__ import annotations

"""
Optional pretrained deep descriptors supplied for the EEE3032 coursework.

The networks are *frozen feature extractors*: students are not expected to train
or fine-tune them.  The first use downloads the standard ImageNet weights from
PyTorch unless those weights are already cached on the machine.

Two deliberately small models are provided:
  * MobileNetV3-Small -> 576-D descriptor
  * SqueezeNet 1.1    -> 512-D descriptor

These functions are optional.  Credit comes from the experimental question,
evaluation and interpretation, not simply from calling a supplied extractor.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np


def _require_torchvision():
    try:
        import torch
        import torchvision
    except ImportError as exc:
        raise ImportError(
            "The optional deep descriptors require torch and torchvision. "
            "In Google Colab they are normally preinstalled.  In another "
            "environment install the packages listed in requirements-deep.txt."
        ) from exc
    return torch, torchvision


def deep_dependencies_available() -> bool:
    try:
        _require_torchvision()
        return True
    except ImportError:
        return False


@dataclass
class DeepDescriptorExtractor:
    """Callable frozen network feature extractor compatible with compute_descriptors()."""

    name: str
    device: str | None = None
    l2_normalize: bool = False

    def __post_init__(self):
        torch, torchvision = _require_torchvision()
        self._torch = torch

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        key = self.name.lower().replace("-", "_")
        if key in {"mobilenet_v3_small", "mobilenet", "mobile"}:
            weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
            model = torchvision.models.mobilenet_v3_small(weights=weights)
            self._preprocess = weights.transforms()
            self._kind = "mobilenet_v3_small"
            self.output_dim = 576
        elif key in {"squeezenet1_1", "squeezenet", "squeeze"}:
            weights = torchvision.models.SqueezeNet1_1_Weights.DEFAULT
            model = torchvision.models.squeezenet1_1(weights=weights)
            self._preprocess = weights.transforms()
            self._kind = "squeezenet1_1"
            self.output_dim = 512
        else:
            raise ValueError(
                "Unknown deep descriptor. Choose 'mobilenet_v3_small' or 'squeezenet1_1'."
            )

        model.eval()
        model.to(self.device)
        self._model = model

    def __call__(self, image_rgb: np.ndarray) -> np.ndarray:
        torch = self._torch
        image = np.asarray(image_rgb)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected an RGB image with shape H x W x 3.")

        # torchvision's standard pretrained transforms accept a CHW uint8 tensor.
        x = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
        x = self._preprocess(x).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            if self._kind == "mobilenet_v3_small":
                z = self._model.features(x)
                z = self._model.avgpool(z)
                z = torch.flatten(z, 1)
            else:  # SqueezeNet 1.1
                z = self._model.features(x)
                z = torch.nn.functional.adaptive_avg_pool2d(z, (1, 1))
                z = torch.flatten(z, 1)

        feature = z.squeeze(0).detach().cpu().numpy().astype(np.float32)
        if self.l2_normalize:
            norm = float(np.linalg.norm(feature))
            if norm > 0:
                feature = feature / norm
        return feature


def make_mobilenet_v3_small_extractor(
    *, device: str | None = None, l2_normalize: bool = False
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a frozen 576-D MobileNetV3-Small feature extractor."""
    return DeepDescriptorExtractor(
        "mobilenet_v3_small", device=device, l2_normalize=l2_normalize
    )


def make_squeezenet1_1_extractor(
    *, device: str | None = None, l2_normalize: bool = False
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a frozen 512-D SqueezeNet-1.1 feature extractor."""
    return DeepDescriptorExtractor(
        "squeezenet1_1", device=device, l2_normalize=l2_normalize
    )



def save_feature_cache(path, records, features, *, descriptor_name: str):
    """Save descriptors with filenames so a cache can be safely re-ordered later."""
    from pathlib import Path
    features = np.asarray(features, dtype=np.float32)
    filenames = np.asarray([r.filename for r in records])
    if len(filenames) != len(features):
        raise ValueError("Number of records and feature rows differs.")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, filenames=filenames, features=features, descriptor_name=descriptor_name)


def load_feature_cache(path, records) -> np.ndarray:
    """Load cached descriptors and return rows ordered to match *records*."""
    data = np.load(path, allow_pickle=False)
    filenames = [str(x) for x in data["filenames"]]
    features = np.asarray(data["features"], dtype=np.float32)
    if len(filenames) != len(features):
        raise ValueError("Invalid feature cache: filenames/features length mismatch.")
    lookup = {name: features[i] for i, name in enumerate(filenames)}
    missing = [r.filename for r in records if r.filename not in lookup]
    if missing:
        raise KeyError(f"Cache is missing {len(missing)} requested images; first: {missing[:3]}")
    return np.vstack([lookup[r.filename] for r in records])
