"""Create canonical arrays/manifests from gprMax outputs and author labels."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np

from u3dgpr.data import (
    CANONICAL_SHAPE,
    bandpass_filter,
    remove_background,
    resize_volume,
)


def _natural_key(path: Path) -> tuple:
    import re

    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name))


def _read_merged_output(path: Path, component: str) -> list[np.ndarray]:
    with h5py.File(path, "r") as handle:
        receivers = handle["rxs"]
        receiver_keys = sorted(receivers.keys(), key=lambda value: int(value.replace("rx", "")))
        traces = []
        for receiver_key in receiver_keys:
            data = np.asarray(receivers[receiver_key][component])
            if data.ndim == 1:
                data = data[None, :]
            traces.append(data)
    return traces


def read_gprmax_volume(path: Path, component: str = "Ez") -> np.ndarray:
    """Read 20 merged channel files or one file containing 20 receivers.

    A standard gprMax ``-n 54`` run is merged per antenna channel and stores a
    matrix [54, time] below ``/rxs/rx1/<component>``. Point ``input`` in the
    pairs CSV to the directory containing the 20 merged ``.out`` files.
    """
    files = sorted(path.glob("*.out"), key=_natural_key) if path.is_dir() else [path]
    scans: list[np.ndarray] = []
    for output_file in files:
        scans.extend(_read_merged_output(output_file, component))
    if not scans:
        raise ValueError(f"No receiver traces found in {path}")
    survey_sizes = {scan.shape[0] for scan in scans}
    time_sizes = {scan.shape[1] for scan in scans}
    if len(survey_sizes) != 1 or len(time_sizes) != 1:
        raise ValueError(f"Inconsistent receiver shapes in {path}")
    return np.stack(scans, axis=0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True, help="CSV columns: id,input,target,target_type,split")
    parser.add_argument("--output", default="prepared_data")
    parser.add_argument("--component", default="Ez")
    parser.add_argument("--skip-background-removal", action="store_true")
    parser.add_argument("--background-sample-count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--center-frequency-hz", type=float, default=None)
    parser.add_argument("--dt-seconds", type=float, default=None)
    args = parser.parse_args()

    with Path(args.pairs).open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    background = None
    if not args.skip_background_removal:
        # Select 500 A-scans uniformly over the corpus without retaining all
        # 1036 paper volumes in RAM. Random-key top-k is equivalent to sampling
        # without replacement from the concatenated trace population.
        rng = np.random.default_rng(args.seed)
        selected_keys = np.empty((0,), dtype=np.float64)
        selected_traces = np.empty((0, CANONICAL_SHAPE[-1]), dtype=np.float32)
        for row in rows:
            volume = resize_volume(read_gprmax_volume(Path(row["input"]), args.component))
            traces = volume.reshape(-1, volume.shape[-1])
            keys = rng.random(traces.shape[0])
            selected_keys = np.concatenate((selected_keys, keys))
            selected_traces = np.concatenate((selected_traces, traces), axis=0)
            keep = min(args.background_sample_count, selected_keys.size)
            indices = np.argpartition(selected_keys, keep - 1)[:keep]
            selected_keys = selected_keys[indices]
            selected_traces = selected_traces[indices]
        background = selected_traces.mean(axis=0, dtype=np.float64).astype(np.float32)
    manifests: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}
    for row in rows:
        volume = resize_volume(read_gprmax_volume(Path(row["input"]), args.component))
        sample_id = row["id"]
        target_values = np.loadtxt(row["target"], dtype=np.float32).reshape(CANONICAL_SHAPE)
        if background is not None:
            volume = remove_background(volume, background)
        if args.center_frequency_hz is not None:
            if args.dt_seconds is None:
                raise ValueError("--dt-seconds is required when band-pass filtering real data")
            volume = bandpass_filter(volume, args.dt_seconds, args.center_frequency_hz)
        sample_path = (output_root / f"{sample_id}.npz").resolve()
        np.savez_compressed(sample_path, gpr=volume, permittivity=target_values)
        split = row["split"]
        manifests[split].append(
            {"id": sample_id, "input": str(sample_path), "target": str(sample_path), "target_type": row["target_type"]}
        )
    for split, split_rows in manifests.items():
        with (output_root / f"{split}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "input", "target", "target_type"])
            writer.writeheader()
            writer.writerows(split_rows)


if __name__ == "__main__":
    main()
