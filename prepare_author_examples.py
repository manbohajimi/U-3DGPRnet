"""Assemble the author's public B-scan PNG examples into 3-D GPR volumes.

Each scene directory contains 20 ``*_B.png`` radar profiles and one flattened
``undergroundTarget.txt`` permittivity label.  A scene, not an individual PNG,
is one training example.  The output axis order is [channel, survey, time] =
[20, 54, 512], matching Section 2.1 and Fig. 2 of Li et al.

The PNG files are display-rendered, lossy examples.  They are useful for
pipeline validation and tiny-data overfitting checks, but paper-level metrics
require regenerated/raw gprMax amplitudes.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
from skimage.io import imread
from skimage.transform import resize

from u3dgpr.data import CANONICAL_SHAPE


def natural_key(path: Path) -> tuple[object, ...]:
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name))


def read_bscan(path: Path) -> np.ndarray:
    image = np.asarray(imread(path))
    source_dtype = image.dtype
    if image.ndim == 3:
        rgb = image[..., :3].astype(np.float32)
        image = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    image = image.astype(np.float32)
    if np.issubdtype(source_dtype, np.integer):
        image /= float(np.iinfo(source_dtype).max)

    # Raster convention is [time, survey]. Network convention is
    # [survey, time]. Bilinear resizing follows Section 2.1.
    resized = resize(
        image,
        output_shape=(CANONICAL_SHAPE[2], CANONICAL_SHAPE[1]),
        order=1,
        mode="reflect",
        anti_aliasing=True,
        preserve_range=True,
    )
    return np.asarray(resized.T, dtype=np.float32)


def read_scene(scene: Path) -> tuple[np.ndarray, np.ndarray]:
    profiles = sorted(scene.glob("*_B.png"), key=natural_key)
    expected_names = [f"{index}_B.png" for index in range(CANONICAL_SHAPE[0])]
    if [path.name for path in profiles] != expected_names:
        raise ValueError(f"{scene}: expected exactly 0_B.png ... 19_B.png")
    volume = np.stack([read_bscan(path) for path in profiles]).astype(np.float32)
    target_path = scene / "undergroundTarget.txt"
    target = np.loadtxt(target_path, dtype=np.float32)
    if target.size != int(np.prod(CANONICAL_SHAPE)):
        raise ValueError(
            f"{target_path}: expected {np.prod(CANONICAL_SHAPE)} values, got {target.size}"
        )
    return volume, target.reshape(CANONICAL_SHAPE)


def target_type(scene: Path) -> str:
    parent = scene.parent.name.lower()
    if "cavity" in parent:
        return "cavity"
    if "pipeline" in parent:
        return "pipeline"
    if "undergroundlayer" in parent:
        return "underground_layer"
    return "unknown"


def split_scenes(scenes: list[Path], seed: int) -> dict[Path, str]:
    """Create a deterministic, class-stratified demo split.

    These splits are not the paper's 836/50/150 split because only a small
    public subset is available.
    """

    rng = np.random.default_rng(seed)
    assignments: dict[Path, str] = {}
    for category in sorted({target_type(scene) for scene in scenes}):
        group = [scene for scene in scenes if target_type(scene) == category]
        rng.shuffle(group)
        validation_count = 1 if len(group) >= 3 else 0
        test_count = 2 if len(group) >= 6 else (1 if len(group) >= 2 else 0)
        for index, scene in enumerate(group):
            if index < validation_count:
                split = "val"
            elif index < validation_count + test_count:
                split = "test"
            else:
                split = "train"
            assignments[scene] = split
    return assignments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Author repository root")
    parser.add_argument("--output", default="prepared_author_examples")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    scenes = sorted(
        {path.parent for path in root.rglob("undergroundTarget.txt") if list(path.parent.glob("*_B.png"))},
        key=lambda path: str(path).lower(),
    )
    if not scenes:
        raise ValueError(f"No author example scenes found below {root}")
    assignments = split_scenes(scenes, args.seed)
    manifests: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}

    for scene in scenes:
        volume, target = read_scene(scene)
        category = target_type(scene)
        sample_id = f"{category}_{scene.name}"
        sample_path = output / f"{sample_id}.npz"
        np.savez_compressed(sample_path, gpr=volume, permittivity=target)
        split = assignments[scene]
        manifests[split].append(
            {
                "id": sample_id,
                "input": str(sample_path),
                "target": str(sample_path),
                "target_type": category,
            }
        )
        print(f"{split:5s} {sample_id}: gpr={volume.shape} target={target.shape}")

    for split, rows in manifests.items():
        manifest_path = output / f"{split}.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "input", "target", "target_type"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {manifest_path} ({len(rows)} scenes)")

    print("WARNING: public PNGs are lossy examples; use raw/regenerated gprMax data for quantitative reproduction.")


if __name__ == "__main__":
    main()
