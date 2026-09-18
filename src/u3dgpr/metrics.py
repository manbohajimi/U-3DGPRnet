"""Metrics reported in Sections 3.2 and 4.2."""

from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity


TARGET_RANGES = {
    "cavity": (1.0, 5.0),
    "underground_layer": (10.0, 15.0),
    "metal_pipeline": (1.0, 8.0),
    "nonmetal_pipeline": (1.0, 5.0),
}


def mse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((np.asarray(prediction) - np.asarray(target)) ** 2))


def ssim3d(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    data_range = max(float(target.max() - target.min()), 1e-6)
    return float(structural_similarity(target, prediction, data_range=data_range))


def threshold_mask(volume: np.ndarray, target_type: str) -> np.ndarray:
    if target_type not in TARGET_RANGES:
        raise ValueError(f"Unknown target type {target_type!r}; choose from {tuple(TARGET_RANGES)}")
    low, high = TARGET_RANGES[target_type]
    volume = np.asarray(volume)
    return (volume >= low) & (volume <= high)


def iou(prediction: np.ndarray, target: np.ndarray, target_type: str) -> float:
    pred_mask = threshold_mask(prediction, target_type)
    target_mask = threshold_mask(target, target_type)
    union = np.logical_or(pred_mask, target_mask).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred_mask, target_mask).sum() / union)

