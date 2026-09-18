from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader

from u3dgpr.config import load_config
from u3dgpr.data import GPRVolumeDataset
from u3dgpr.engine import (
    DEFAULT_BACKGROUND_VALUE,
    DEFAULT_TARGET_THRESHOLD,
    build_loss,
    run_epoch,
    save_checkpoint,
    seed_everything,
)
from u3dgpr.model import U3DGPRNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train U-3DGPR-Net")
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument(
        "--stage",
        choices=("vertical", "channel_crossed", "full"),
        default=None,
        help="Paper-style staged training: vertical, channel_crossed, then full fusion/refinement",
    )
    parser.add_argument("--pretrained", default=None, help="Checkpoint from the preceding stage")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--foreground-weight", type=float, default=None)
    parser.add_argument("--background-weight", type=float, default=None)
    return parser.parse_args()


def make_dataset(path: str, data_config: dict) -> GPRVolumeDataset:
    return GPRVolumeDataset(
        path,
        shape=tuple(data_config["shape"]),
        input_normalization=data_config["input_normalization"],
        target_scale=data_config["target_scale"],
        axis_order=tuple(data_config.get("axis_order", (0, 1, 2))),
        input_min=float(data_config.get("input_min", -9.0)),
        input_max=float(data_config.get("input_max", 9.0)),
    )


def validate_manifest_splits(data_config: dict) -> None:
    """Fail fast on duplicate scenes within or across benchmark splits.

    ``data.allow_val_test_overlap`` defaults to False, so the cross-split check
    keeps its teeth everywhere else. It exists for the 3DInvNet upstream
    protocol, where the released 150-scene test set doubles as the validation
    set: training every model on one identical split matters more there than the
    overlap, which is a property of the published dataset rather than of our
    pipeline. Enabling it downgrades the cross-split hit to a loud warning.
    """

    allow_val_test_overlap = bool(data_config.get("allow_val_test_overlap", False))
    split_keys: dict[str, set[tuple[str, str]]] = {}
    for split in ("train", "val", "test"):
        manifest = data_config.get(f"{split}_manifest")
        if not manifest:
            continue
        with Path(manifest).open("r", newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        keys = {(row["input"], row["target"]) for row in rows}
        if len(keys) != len(rows):
            raise ValueError(f"Duplicate scenes inside {split} manifest: {manifest}")
        split_keys[split] = keys

    names = tuple(split_keys)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = split_keys[left] & split_keys[right]
            if overlap:
                example = next(iter(overlap))
                message = (
                    f"Data leakage: {len(overlap)} scene(s) shared by {left} and {right}; "
                    f"example input={example[0]!r}"
                )
                if not allow_val_test_overlap:
                    raise ValueError(message)
                print(f"WARNING: {message}", flush=True)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(args.seed if args.seed is not None else config["seed"]))
    data_config = config["data"]
    validate_manifest_splits(data_config)
    train_dataset = make_dataset(data_config["train_manifest"], data_config)
    if data_config.get("simulated_manifest") and data_config.get("real_manifest"):
        train_dataset = ConcatDataset(
            [make_dataset(data_config["simulated_manifest"], data_config), make_dataset(data_config["real_manifest"], data_config)]
        )
    val_dataset = make_dataset(data_config["val_manifest"], data_config)
    train_config = config["train"]
    loader_options = {
        "batch_size": int(train_config["batch_size"]),
        "num_workers": int(data_config["num_workers"]),
        "pin_memory": True,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)

    requested = config["device"]
    device = torch.device("cuda" if requested == "cuda" and torch.cuda.is_available() else "cpu")
    model_kwargs = {k: v for k, v in config["model"].items() if k != "return_intermediates"}
    if args.stage is not None:
        model_kwargs["output_mode"] = args.stage
    model = U3DGPRNet(**model_kwargs).to(device)
    pretrained = args.pretrained or train_config.get("pretrained")
    if pretrained:
        checkpoint = torch.load(pretrained, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint.get("model", checkpoint))

    stage = args.stage or str(model_kwargs.get("output_mode", "full"))
    if stage == "vertical":
        trainable_parameters = list(model.vertical_branch.parameters())
    elif stage == "channel_crossed":
        trainable_parameters = list(model.channel_branch.parameters())
    else:
        trainable_parameters = list(model.parameters())
    optimizer = torch.optim.Adamax(
        trainable_parameters,
        lr=float(
            args.learning_rate
            if args.learning_rate is not None
            else train_config["learning_rate"]
        ),
        weight_decay=float(train_config["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(train_config["amp"]) and device.type == "cuda")

    # 稀疏目标损失配置。不同 loss 只读取自己需要的字段。
    loss_config = {
        "background_value": float(train_config.get("background_value", DEFAULT_BACKGROUND_VALUE)),
        "target_threshold": float(train_config.get("target_threshold", DEFAULT_TARGET_THRESHOLD)),
        "foreground_weight": float(
            args.foreground_weight
            if args.foreground_weight is not None
            else train_config.get("foreground_weight", 1.0)
        ),
        "background_weight": float(
            args.background_weight
            if args.background_weight is not None
            else train_config.get("background_weight", 1.0)
        ),
    }
    criterion = build_loss(train_config.get("loss", "mse"), **loss_config)
    # 区域指标只需要阈值；验证损失本身必须继续使用同一 criterion。
    region_kwargs = {
        "background_value": loss_config["background_value"],
        "target_threshold": loss_config["target_threshold"],
        "mask_threshold": float(train_config.get("mask_threshold", 6.0)),
    }
    select_metric = str(train_config.get("select_metric", "loss"))
    select_mode = str(
        train_config.get(
            "select_mode",
            "max" if select_metric in {"iou", "dice", "precision", "recall"} else "min",
        )
    )
    if select_mode not in {"min", "max"}:
        raise ValueError("train.select_mode must be 'min' or 'max'")
    print(
        f"stage={stage} loss={train_config.get('loss', 'mse')} "
        f"select_metric={select_metric} select_mode={select_mode} trainable_parameters="
        f"{sum(parameter.numel() for parameter in trainable_parameters):,} {loss_config}",
        flush=True,
    )

    output_dir = Path(args.output_dir or config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.csv"
    best_score = float("inf") if select_mode == "min" else float("-inf")
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "val_loss",
                "val_mse",
                "val_target_mse",
                "val_background_mse",
                "val_target_mae",
                "val_background_mae",
                "val_iou",
                "val_dice",
                "val_precision",
                "val_recall",
            ]
        )
        epoch_count = int(args.epochs if args.epochs is not None else train_config["epochs"])
        for epoch in range(1, epoch_count + 1):
            train_metrics = run_epoch(
                model, train_loader, device, optimizer, scaler, bool(train_config["amp"]), criterion, **region_kwargs
            )
            val_metrics = run_epoch(
                model, val_loader, device, None, None, bool(train_config["amp"]), criterion, **region_kwargs
            )
            writer.writerow(
                [
                    epoch,
                    train_metrics["loss"],
                    val_metrics["loss"],
                    val_metrics["mse"],
                    val_metrics["target_mse"],
                    val_metrics["background_mse"],
                    val_metrics["target_mae"],
                    val_metrics["background_mae"],
                    val_metrics["iou"],
                    val_metrics["dice"],
                    val_metrics["precision"],
                    val_metrics["recall"],
                ]
            )
            handle.flush()
            print(
                f"epoch={epoch:03d} train_loss={train_metrics['loss']:.6f} val_loss={val_metrics['loss']:.6f} "
                f"val_mse={val_metrics['mse']:.6f} val_target_mse={val_metrics['target_mse']:.4f} "
                f"val_background_mse={val_metrics['background_mse']:.6f} "
                f"val_target_mae={val_metrics['target_mae']:.4f} "
                f"val_background_mae={val_metrics['background_mae']:.4f} "
                f"val_iou={val_metrics['iou']:.4f} val_precision={val_metrics['precision']:.4f} "
                f"val_recall={val_metrics['recall']:.4f}",
                flush=True,
            )
            score = val_metrics.get(select_metric, val_metrics["loss"])
            improved = score < best_score if select_mode == "min" else score > best_score
            if improved:
                best_score = score
                save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, val_metrics)
            if epoch % int(train_config["checkpoint_every"]) == 0:
                save_checkpoint(output_dir / f"epoch_{epoch:03d}.pt", model, optimizer, epoch, val_metrics)


if __name__ == "__main__":
    main()
