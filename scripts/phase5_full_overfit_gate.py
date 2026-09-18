"""Multi-seed single-scene overfit gate for the repaired full U-3DGPR path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from u3dgpr.config import load_config
from u3dgpr.data import GPRVolumeDataset
from u3dgpr.engine import build_loss, region_masks, seed_everything
from u3dgpr.model import U3DGPRNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/transfer_3dinvnet.yaml")
    parser.add_argument(
        "--work-dir",
        default="/home/bqwang/project/GPR/model/temp_code/u3dgpr_benchmark/single_sample",
        help="Directory containing train.csv and probe.csv",
    )
    parser.add_argument("--output", default="phase5_full_overfit_gate.json")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 42, 3407))
    parser.add_argument("--max-background-mae", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_config = config["data"]
    train_config = config["train"]
    dataset_kwargs = {
        "shape": tuple(data_config["shape"]),
        "input_normalization": data_config["input_normalization"],
        "target_scale": data_config["target_scale"],
        "axis_order": tuple(data_config.get("axis_order", (0, 1, 2))),
    }
    work_dir = Path(args.work_dir)
    one = GPRVolumeDataset(work_dir / "train.csv", **dataset_kwargs)[0]
    probe = GPRVolumeDataset(work_dir / "probe.csv", **dataset_kwargs)[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpr = one["gpr"][None].to(device)
    target = one["target"][None].to(device)
    other_gpr = probe["gpr"][None].to(device)

    background_value = float(train_config.get("background_value", 4.0))
    target_threshold = float(train_config.get("target_threshold", 1e-6))
    criterion = build_loss(
        train_config.get("loss", "balanced_mse"),
        background_value=background_value,
        target_threshold=target_threshold,
        foreground_weight=float(train_config.get("foreground_weight", 1.0)),
        background_weight=float(train_config.get("background_weight", 1.0)),
    )
    foreground, background = region_masks(target, background_value, target_threshold)
    baseline_mse = float((target - background_value).square().mean())
    model_kwargs = {k: v for k, v in config["model"].items() if k != "return_intermediates"}
    model_kwargs["output_mode"] = "full"

    results: dict[str, object] = {
        "sample": one["id"],
        "probe": probe["id"],
        "constant_background_mse": baseline_mse,
        "runs": {},
    }
    all_passed = True
    for seed in args.seeds:
        seed_everything(seed)
        model = U3DGPRNet(**model_kwargs).to(device)
        optimizer = torch.optim.Adamax(
            model.parameters(),
            lr=float(train_config["learning_rate"]),
            weight_decay=float(train_config["weight_decay"]),
        )
        use_amp = bool(train_config["amp"]) and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        rows: list[dict[str, float | int | None]] = []

        def snapshot(step: int, train_loss: float | None) -> dict[str, float | int | None]:
            model.eval()
            with torch.inference_mode():
                prediction = model(gpr).reconstruction
                other_prediction = model(other_gpr).reconstruction
                measured_loss = float(criterion(prediction, target))
            model.train()
            error = prediction - target
            return {
                "step": step,
                "loss": measured_loss,
                "train_loss": train_loss,
                "mse": float(error.square().mean()),
                "target_mse": float(error.square()[foreground].mean()),
                "background_mse": float(error.square()[background].mean()),
                "target_mae": float(error.abs()[foreground].mean()),
                "background_mae": float(error.abs()[background].mean()),
                "pred_std": float(prediction.std()),
                "pred_min": float(prediction.min()),
                "pred_max": float(prediction.max()),
                "input_difference": float((prediction - other_prediction).abs().max()),
            }

        initial = snapshot(0, None)
        rows.append(initial)
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(gpr).reconstruction, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if step % args.log_every == 0 or step == args.steps:
                row = snapshot(step, float(loss.detach()))
                rows.append(row)
                print(
                    f"seed={seed} step={step:4d} loss={row['loss']:.5f} "
                    f"mse={row['mse']:.5f} target_mae={row['target_mae']:.4f} "
                    f"std={row['pred_std']:.4f}",
                    flush=True,
                )

        final = rows[-1]
        # This gate answers whether the repaired path can learn without
        # collapsing. Full-volume MSE is retained as a diagnostic: the model
        # starts at the constant-background MSE optimum, while balanced loss
        # deliberately gives the sparse foreground much more influence.
        checks = {
            "loss_decreased_50pct": final["loss"] < initial["loss"] * 0.5,
            "target_mae_decreased_20pct": final["target_mae"] < initial["target_mae"] * 0.8,
            "not_constant": final["pred_std"] > 1e-3,
            "input_dependent": final["input_difference"] > 1e-4,
            "background_mae_bounded": final["background_mae"] < args.max_background_mae,
        }
        diagnostics = {
            "beats_constant_background_mse": final["mse"] < baseline_mse,
            "target_mae_below_2": final["target_mae"] < 2.0,
            "physical_min_at_least_1": final["pred_min"] >= 1.0,
        }
        passed = all(checks.values())
        all_passed = all_passed and passed
        results["runs"][str(seed)] = {
            "passed": passed,
            "checks": checks,
            "diagnostics": diagnostics,
            "history": rows,
        }
        print(
            f"seed={seed} passed={passed} checks={checks} diagnostics={diagnostics}",
            flush=True,
        )
        del model, optimizer, scaler
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results["passed"] = all_passed
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {output}; gate_passed={all_passed}")
    raise SystemExit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
