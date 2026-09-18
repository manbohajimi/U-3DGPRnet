from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from u3dgpr.data import load_array
from u3dgpr.metrics import iou, mse, ssim3d


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved permittivity volumes")
    parser.add_argument("--manifest", required=True, help="CSV: id,prediction,target,target_type")
    parser.add_argument("--output", default="metrics.json")
    args = parser.parse_args()
    with Path(args.manifest).open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    results = []
    for row in rows:
        prediction = load_array(row["prediction"], key="permittivity")
        target = load_array(row["target"], key="permittivity")
        results.append(
            {
                "id": row["id"],
                "mse": mse(prediction, target),
                "ssim": ssim3d(prediction, target),
                "iou": iou(prediction, target, row["target_type"]),
            }
        )
    summary = {
        "samples": results,
        "mean": {key: float(np.mean([item[key] for item in results])) for key in ("mse", "ssim", "iou")},
    }
    Path(args.output).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary["mean"], indent=2))


if __name__ == "__main__":
    main()

