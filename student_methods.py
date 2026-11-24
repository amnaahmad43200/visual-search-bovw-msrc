"""
EEE3032 coursework: put your own descriptor / distance / representation ideas here.

The supplied global RGB histogram is only the baseline. Two frozen deep descriptor extractors are supplied only as optional extension tools.
Your core work should answer one focused research question using a representation or
retrieval method that you design or modify yourself. Test it using the fixed development
and test protocol.
"""

import numpy as np


def extract_descriptor(image_rgb: np.ndarray) -> np.ndarray:
    """
    Replace this with one of your investigated descriptors.

    Requirements:
      * input: H x W x 3 RGB uint8 image
      * output: one fixed-length 1-D numpy array
    """
    raise NotImplementedError("Implement and justify your descriptor here.")


def distance(query_feature: np.ndarray, gallery_features: np.ndarray) -> np.ndarray:
    """
    Optional: replace this with your chosen distance/similarity measure.

    It must return one numeric distance per gallery image; smaller means more similar.
    """
    q = np.asarray(query_feature, dtype=np.float32).ravel()
    g = np.asarray(gallery_features, dtype=np.float32)
    return np.linalg.norm(g - q[None, :], axis=1)
