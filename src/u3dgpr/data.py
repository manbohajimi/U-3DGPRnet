"""Dataset and preprocessing utilities for canonical 3-D GPR volumes."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.signal import butter, sosfiltfilt


CANONICAL_SHAPE = (20, 54, 512)


def load_array(path: str | Path, key: str | None = None) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".npz":
        archive = np.load(path)
        selected = key or ("gpr" if "gpr" in archive.files else archive.files[0])
        return archive[selected]
    if path.suffix == ".mat":
        # 3DInvNet 仿真数据集：clean_data = GPR 输入，mask = 相对介电常数 GT。
        from scipy.io import loadmat

        variable = {"gpr": "clean_data", "permittivity": "mask"}.get(key)
        if variable is None:
            raise ValueError(f"Unsupported key {key!r} for .mat input: {path}")
        return np.asarray(loadmat(path)[variable], dtype=np.float32)
    if path.suffix == ".txt":
        values = np.loadtxt(path, dtype=np.float32)
        if values.size == np.prod(CANONICAL_SHAPE):
            return values.reshape(CANONICAL_SHAPE)
        return values
    raise ValueError(f"Unsupported array format: {path}")


def resize_volume(array: np.ndarray, shape: tuple[int, int, int] = CANONICAL_SHAPE) -> np.ndarray:
    if tuple(array.shape) == tuple(shape):
        return np.asarray(array, dtype=np.float32)
    # The paper names bilinear rather than trilinear interpolation. Resample
    # the survey-time planes first, then the channel-time planes.
    target_channels, target_survey, target_time = shape
    tensor = torch.as_tensor(np.asarray(array), dtype=torch.float32)
    vertical = F.interpolate(
        tensor[:, None], size=(target_survey, target_time), mode="bilinear", align_corners=False
    )[:, 0]
    crossed = vertical.permute(1, 0, 2)[:, None]
    crossed = F.interpolate(
        crossed, size=(target_channels, target_time), mode="bilinear", align_corners=False
    )[:, 0]
    return crossed.permute(1, 0, 2).contiguous().numpy()


def normalize_gpr(
    array: np.ndarray,
    method: str,
    input_min: float = -9.0,
    input_max: float = 9.0,
) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if method == "none":
        return array
    if method == "zscore":
        return (array - array.mean()) / max(float(array.std()), 1e-6)
    if method == "maxabs":
        return array / max(float(np.abs(array).max()), 1e-6)
    if method == "fixed_minmax":
        if input_max <= input_min:
            raise ValueError("input_max must be greater than input_min")
        # Match the public 3DInvNet input contract exactly: (x + 9) / 18.
        # Unlike per-volume z-score this preserves absolute GPR amplitude,
        # which carries information needed by permittivity regression.
        return (array - float(input_min)) / (float(input_max) - float(input_min))
    raise ValueError(f"Unknown normalization: {method}")


class GPRVolumeDataset(Dataset):
    """CSV manifest with columns: id,input,target and optional target_type."""

    def __init__(
        self,
        manifest: str | Path,
        shape: tuple[int, int, int] = CANONICAL_SHAPE,
        input_normalization: str = "zscore",
        target_scale: float = 1.0,
        axis_order: tuple[int, int, int] | None = None,
        input_min: float = -9.0,
        input_max: float = 9.0,
    ):
        self.manifest = Path(manifest)
        self.shape = tuple(shape)
        self.input_normalization = input_normalization
        self.target_scale = float(target_scale)
        self.axis_order = tuple(axis_order) if axis_order is not None else (0, 1, 2)
        self.input_min = float(input_min)
        self.input_max = float(input_max)
        if sorted(self.axis_order) != [0, 1, 2]:
            raise ValueError(f"axis_order must be a permutation of [0, 1, 2], got {self.axis_order}")
        with self.manifest.open("r", newline="", encoding="utf-8-sig") as handle:
            self.rows = list(csv.DictReader(handle))
        if not self.rows:
            raise ValueError(f"Empty manifest: {self.manifest}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        row = self.rows[index]
        gpr = load_array(row["input"], key="gpr")
        target = load_array(row["target"], key="permittivity")
        # Convert source storage order to model order [channel, survey, time].
        # This must happen before resize; cubic 128^3 arrays otherwise conceal
        # an incorrect axis interpretation because their shapes do not change.
        gpr = np.transpose(gpr, self.axis_order)
        target = np.transpose(target, self.axis_order)
        gpr = normalize_gpr(
            resize_volume(gpr, self.shape),
            self.input_normalization,
            input_min=self.input_min,
            input_max=self.input_max,
        )
        target = resize_volume(target, self.shape) / self.target_scale
        return {
            "id": row.get("id", str(index)),
            "gpr": torch.from_numpy(gpr).unsqueeze(0),
            "target": torch.from_numpy(target).unsqueeze(0),
            "target_type": row.get("target_type", "unknown"),
        }


def estimate_background(volumes: np.ndarray, sample_count: int = 500, seed: int = 42) -> np.ndarray:
    """Mean of randomly selected A-scans, following Section 3.1."""
    if volumes.ndim != 4:
        raise ValueError("Expected [N, channels, survey, time]")
    traces = volumes.reshape(-1, volumes.shape[-1])
    count = min(sample_count, traces.shape[0])
    indices = np.random.default_rng(seed).choice(traces.shape[0], count, replace=False)
    return traces[indices].mean(axis=0, dtype=np.float64).astype(np.float32)


def remove_background(volume: np.ndarray, background: np.ndarray) -> np.ndarray:
    return np.asarray(volume, dtype=np.float32) - np.asarray(background, dtype=np.float32)[None, None, :]


def bandpass_filter(volume: np.ndarray, dt_seconds: float, center_frequency_hz: float, order: int = 4) -> np.ndarray:
    """Real-data filter from Section 4.1: pass band [0.5 fc, 2 fc]."""
    sampling_frequency = 1.0 / float(dt_seconds)
    nyquist = sampling_frequency / 2.0
    low = 0.5 * float(center_frequency_hz)
    high = min(2.0 * float(center_frequency_hz), nyquist * 0.999)
    if not 0.0 < low < high < nyquist:
        raise ValueError(
            f"Invalid band [{low}, {high}] Hz for Nyquist frequency {nyquist} Hz"
        )
    sos = butter(order, [low, high], btype="bandpass", fs=sampling_frequency, output="sos")
    return sosfiltfilt(sos, np.asarray(volume, dtype=np.float32), axis=-1).astype(np.float32)
