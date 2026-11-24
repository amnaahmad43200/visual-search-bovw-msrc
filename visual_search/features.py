"""Feature extraction and descriptor transformations used in the coursework."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

SIFT_DIMENSION = 128
OPPONENT_SIFT_DIMENSION = 3 * SIFT_DIMENSION


def create_sift():
    """Create OpenCV's standard SIFT extractor with its default parameters."""
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError("This OpenCV installation does not provide SIFT_create().")
    return cv2.SIFT_create()


def _check_rgb_image(image_rgb: np.ndarray) -> np.ndarray:
    """Validate an RGB image and return it as uint8."""
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape (height, width, 3).")
    if image.dtype != np.uint8:
        if not np.isfinite(image).all() or image.min() < 0 or image.max() > 255:
            raise ValueError("RGB values must be finite and in the range [0, 255].")
        image = np.rint(image).astype(np.uint8)
    return image


def _empty_descriptors(dimension: int) -> np.ndarray:
    """Return a correctly shaped empty descriptor matrix."""
    return np.empty((0, dimension), dtype=np.float32)


def keypoints_to_array(keypoints: Sequence[cv2.KeyPoint], include_details: bool = True) -> np.ndarray:
    """Store keypoint properties in a numeric array so they can be cached."""
    if include_details:
        rows = [
            [kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id]
            for kp in keypoints
        ]
        return np.asarray(rows, dtype=np.float32).reshape(-1, 7)

    rows = [[kp.pt[0], kp.pt[1], kp.size, kp.angle] for kp in keypoints]
    return np.asarray(rows, dtype=np.float32).reshape(-1, 4)


def keypoints_from_array(keypoint_info: np.ndarray) -> list[cv2.KeyPoint]:
    """Recreate OpenCV KeyPoint objects from cached keypoint properties."""
    info = np.asarray(keypoint_info, dtype=np.float32)
    if info.ndim != 2 or info.shape[1] not in (4, 7):
        raise ValueError("keypoint_info must have either 4 or 7 columns.")

    keypoints = []
    for row in info:
        point = cv2.KeyPoint(float(row[0]), float(row[1]), float(row[2]), float(row[3]))
        if info.shape[1] == 7:
            point.response = float(row[4])
            point.octave = int(row[5])
            point.class_id = int(row[6])
        keypoints.append(point)
    return keypoints


def extract_detected_sift(image_rgb: np.ndarray, sift=None) -> tuple[np.ndarray, np.ndarray]:
    """Detect SIFT keypoints automatically and return keypoint data and descriptors."""
    image = _check_rgb_image(image_rgb)
    extractor = sift if sift is not None else create_sift()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    keypoints, descriptors = extractor.detectAndCompute(gray, None)

    if descriptors is None:
        descriptors = _empty_descriptors(SIFT_DIMENSION)
    return keypoints_to_array(keypoints, include_details=True), descriptors.astype(np.float32)


def make_dense_keypoints(
    image_shape: tuple[int, int] | np.ndarray,
    step: int = 12,
    keypoint_size: float = 16,
    angle: float = 0,
) -> list[cv2.KeyPoint]:
    """Place keypoints on a regular grid for dense SIFT extraction.

    `image_shape` can be a grayscale image or its (height, width) shape.
    Grid points are kept half a patch width away from the image boundary.
    """
    if isinstance(image_shape, np.ndarray):
        height, width = image_shape.shape[:2]
    else:
        height, width = image_shape[:2]
    if step <= 0 or keypoint_size <= 0:
        raise ValueError("step and keypoint_size must be positive.")

    margin = int(keypoint_size / 2)
    points = []
    for y in range(margin, height - margin, step):
        for x in range(margin, width - margin, step):
            points.append(cv2.KeyPoint(float(x), float(y), float(keypoint_size), float(angle)))
    return points


def extract_dense_sift(
    image_rgb: np.ndarray,
    sift=None,
    step: int = 12,
    keypoint_size: float = 16,
    angle: float = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate SIFT descriptors at regularly spaced locations."""
    image = _check_rgb_image(image_rgb)
    extractor = sift if sift is not None else create_sift()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    grid_points = make_dense_keypoints(gray, step, keypoint_size, angle)
    keypoints, descriptors = extractor.compute(gray, grid_points)

    if descriptors is None:
        descriptors = _empty_descriptors(SIFT_DIMENSION)
    return keypoints_to_array(keypoints, include_details=False), descriptors.astype(np.float32)


def root_sift(descriptors: np.ndarray) -> np.ndarray:
    """Apply RootSIFT row-wise: L1-normalise, then take the square root."""
    values = np.asarray(descriptors, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("Descriptors must be a 2-D array.")
    if not np.isfinite(values).all():
        raise ValueError("Descriptors contain NaN or infinite values.")
    if np.any(values < 0):
        raise ValueError("SIFT descriptors must be non-negative for RootSIFT.")
    if values.shape[0] == 0:
        return values.copy()

    row_sums = values.sum(axis=1, keepdims=True)
    normalised = np.divide(
        values,
        row_sums,
        out=np.zeros_like(values),
        where=row_sums > 0,
    )
    return np.sqrt(normalised).astype(np.float32)


def opponent_channels(image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert RGB into three fixed-range uint8 opponent-colour channels.

    O1 represents red-green opposition, O2 represents yellow-blue opposition,
    and O3 represents overall intensity. Fixed theoretical ranges are used for
    every image; channels are not contrast-stretched individually per image.
    """
    image = _check_rgb_image(image_rgb).astype(np.float32)
    red, green, blue = image[..., 0], image[..., 1], image[..., 2]

    o1 = (red - green) / np.sqrt(2.0)
    o2 = (red + green - 2.0 * blue) / np.sqrt(6.0)
    o3 = (red + green + blue) / np.sqrt(3.0)

    o1_limit = 255.0 / np.sqrt(2.0)
    o2_limit = 510.0 / np.sqrt(6.0)
    o3_limit = 765.0 / np.sqrt(3.0)

    # OpenCV SIFT expects an 8-bit image. Signed channels are mapped using
    # fixed limits so the same numerical meaning is retained across images.
    o1_u8 = np.clip((o1 + o1_limit) / (2.0 * o1_limit) * 255.0, 0, 255)
    o2_u8 = np.clip((o2 + o2_limit) / (2.0 * o2_limit) * 255.0, 0, 255)
    o3_u8 = np.clip(o3 / o3_limit * 255.0, 0, 255)

    return tuple(np.rint(channel).astype(np.uint8) for channel in (o1_u8, o2_u8, o3_u8))


def extract_opponent_sift(
    image_rgb: np.ndarray,
    keypoint_info: np.ndarray,
    sift=None,
) -> np.ndarray:
    """Describe the same keypoint locations in O1, O2 and O3, then concatenate.

    Each channel contributes a 128-D SIFT descriptor. The concatenated
    OpponentSIFT descriptor is therefore 384-D.
    """
    image = _check_rgb_image(image_rgb)
    extractor = sift if sift is not None else create_sift()
    info = np.asarray(keypoint_info, dtype=np.float32)
    if info.ndim != 2 or info.shape[1] not in (4, 7):
        raise ValueError("keypoint_info must have 4 or 7 columns.")
    if info.shape[0] == 0:
        return _empty_descriptors(OPPONENT_SIFT_DIMENSION)

    channel_descriptors = []
    for channel_index, channel in enumerate(opponent_channels(image), start=1):
        supplied_keypoints = keypoints_from_array(info)
        returned_keypoints, descriptors = extractor.compute(channel, supplied_keypoints)
        if descriptors is None:
            descriptors = _empty_descriptors(SIFT_DIMENSION)

        if len(returned_keypoints) != len(info):
            raise ValueError(
                f"Opponent channel {channel_index} returned {len(returned_keypoints)} "
                f"keypoints; {len(info)} were supplied."
            )
        returned_info = keypoints_to_array(returned_keypoints, include_details=False)
        if not np.allclose(returned_info[:, :4], info[:, :4], atol=1e-3):
            raise ValueError(f"Keypoint locations changed in opponent channel {channel_index}.")
        if descriptors.shape != (len(info), SIFT_DIMENSION):
            raise ValueError(f"Unexpected descriptor shape in opponent channel {channel_index}.")
        channel_descriptors.append(descriptors.astype(np.float32))

    return np.concatenate(channel_descriptors, axis=1).astype(np.float32)
