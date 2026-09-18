"""Create a deterministic validation split without touching the public test set."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    required = {"id", "input", "target"}
    if not required.issubset(fields):
        raise ValueError(f"{path} lacks columns {sorted(required - set(fields))}")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"duplicate id inside {path}")
    return fields, rows


def write_manifest(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-pool", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be between 0 and 1")
    train_fields, pool = read_manifest(Path(args.train_pool))
    test_fields, test = read_manifest(Path(args.test_manifest))
    if train_fields != test_fields:
        raise ValueError("train/test manifest columns differ")
    pool_ids = {row["id"] for row in pool}
    test_ids = {row["id"] for row in test}
    overlap = pool_ids & test_ids
    if overlap:
        raise ValueError(f"official train pool overlaps test by {len(overlap)} ids")

    # Stratify when a useful target_type column exists; otherwise use one group.
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    use_type = "target_type" in train_fields and len(
        {row.get("target_type", "") for row in pool}
    ) > 1
    for row in pool:
        groups[row.get("target_type", "unknown") if use_type else "all"].append(row)

    rng = random.Random(args.seed)
    train: list[dict[str, str]] = []
    val: list[dict[str, str]] = []
    for rows in groups.values():
        shuffled = list(rows)
        rng.shuffle(shuffled)
        count = max(1, round(len(shuffled) * args.val_fraction))
        val.extend(shuffled[:count])
        train.extend(shuffled[count:])
    train.sort(key=lambda row: row["id"])
    val.sort(key=lambda row: row["id"])
    test.sort(key=lambda row: row["id"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(output_dir / "train.csv", train_fields, train)
    write_manifest(output_dir / "val.csv", train_fields, val)
    write_manifest(output_dir / "test.csv", test_fields, test)
    summary = {
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "source_train_pool": str(Path(args.train_pool)),
        "source_test_manifest": str(Path(args.test_manifest)),
        "counts": {"train": len(train), "val": len(val), "test": len(test)},
        "id_intersections": {
            "train_val": len({r["id"] for r in train} & {r["id"] for r in val}),
            "train_test": len({r["id"] for r in train} & test_ids),
            "val_test": len({r["id"] for r in val} & test_ids),
        },
        "target_type_counts": {
            split: dict(Counter(row.get("target_type", "unknown") for row in rows))
            for split, rows in (("train", train), ("val", val), ("test", test))
        },
    }
    (output_dir / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
