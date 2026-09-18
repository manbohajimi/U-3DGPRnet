"""Training and evaluation helpers."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


DEFAULT_BACKGROUND_VALUE = 4.0
DEFAULT_TARGET_THRESHOLD = 1e-6


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def region_masks(target: torch.Tensor, background_value: float, target_threshold: float):
    """返回 (前景, 背景) 布尔掩码；前景 = |target - background_value| > threshold。"""
    foreground = torch.abs(target - background_value) > target_threshold
    return foreground, ~foreground


class BalancedMSELoss(nn.Module):
    """前景/背景分开算 MSE 再等权合并。

    全体素 MSE 在极稀疏目标上被背景主导（本数据集目标仅占约 0.05% 体素），
    其最小值是一张常数背景图，模型会退化成平凡解。本损失对每个样本先分别求
    前景与背景的均方误差再合并，使目标体素获得与背景同量级的梯度权重，并
    避免目标较大的样本压过目标较小的样本。与 3DInvNet 已验证的
    ``BalancedMAE_loss`` 同构，只是把绝对误差换成平方误差（论文式 (6) 用 MSE）。
    """

    def __init__(
        self,
        background_value: float = DEFAULT_BACKGROUND_VALUE,
        target_threshold: float = DEFAULT_TARGET_THRESHOLD,
        foreground_weight: float = 1.0,
        background_weight: float = 1.0,
    ):
        super().__init__()
        if foreground_weight < 0 or background_weight < 0:
            raise ValueError("loss weights must be non-negative")
        if foreground_weight + background_weight == 0:
            raise ValueError("at least one loss weight must be positive")
        self.background_value = float(background_value)
        self.target_threshold = float(target_threshold)
        self.foreground_weight = float(foreground_weight)
        self.background_weight = float(background_weight)

    def forward(self, out: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        target = mask.to(out.device, non_blocking=True)
        squared_error = (out - target) ** 2
        foreground, background = region_masks(target, self.background_value, self.target_threshold)

        reduce_dims = tuple(range(1, squared_error.ndim))
        foreground_count = foreground.sum(dim=reduce_dims)
        background_count = background.sum(dim=reduce_dims)
        foreground_per_sample = (
            (squared_error * foreground).sum(dim=reduce_dims) / foreground_count.clamp_min(1)
        )
        background_per_sample = (
            (squared_error * background).sum(dim=reduce_dims) / background_count.clamp_min(1)
        )

        foreground_valid = foreground_count > 0
        background_valid = background_count > 0
        if foreground_valid.any():
            foreground_loss = foreground_per_sample[foreground_valid].mean()
        else:
            foreground_loss = out.sum() * 0.0
        if background_valid.any():
            background_loss = background_per_sample[background_valid].mean()
        else:
            background_loss = out.sum() * 0.0

        weight_sum = self.foreground_weight + self.background_weight
        return (
            self.foreground_weight * foreground_loss + self.background_weight * background_loss
        ) / weight_sum


def build_loss(name: str, **kwargs) -> nn.Module:
    key = str(name).lower()
    if key == "mse":
        return nn.MSELoss()
    if key == "balanced_mse":
        return BalancedMSELoss(
            background_value=kwargs.get("background_value", DEFAULT_BACKGROUND_VALUE),
            target_threshold=kwargs.get("target_threshold", DEFAULT_TARGET_THRESHOLD),
            foreground_weight=kwargs.get("foreground_weight", 1.0),
            background_weight=kwargs.get("background_weight", 1.0),
        )
    raise ValueError(f"Unknown loss {name!r}; choose from 'mse' or 'balanced_mse'")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    use_amp: bool,
    criterion: nn.Module | None = None,
    background_value: float = DEFAULT_BACKGROUND_VALUE,
    target_threshold: float = DEFAULT_TARGET_THRESHOLD,
    mask_threshold: float = 6.0,
) -> dict[str, float]:
    """跑一个 epoch，返回各指标的样本加权平均。

    ``loss`` 始终是传入的训练目标，训练和验证口径一致；``mse`` 恒为全体素 MSE，
    可直接与常数背景解基线对照；``target_mae`` / ``background_mae`` 分区域统计。
    """
    training = optimizer is not None
    model.train(training)
    if criterion is None:
        criterion = nn.MSELoss()

    totals = {
        "loss": 0.0,
        "mse": 0.0,
        "target_mse": 0.0,
        "background_mse": 0.0,
        "target_mae": 0.0,
        "background_mae": 0.0,
        "iou": 0.0,
        "dice": 0.0,
        "precision": 0.0,
        "recall": 0.0,
    }
    sample_count = 0
    for batch in loader:
        gpr = batch["gpr"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
                prediction = model(gpr).reconstruction
                loss = criterion(prediction, target)
            if training:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        with torch.no_grad():
            residual = (prediction.float() - target.float()) ** 2
            absolute = residual.sqrt()
            foreground, background = region_masks(target, background_value, target_threshold)
            batch_size = gpr.shape[0]
            totals["loss"] += float(loss.detach()) * batch_size
            totals["mse"] += float(residual.mean()) * batch_size
            if foreground.any():
                totals["target_mse"] += float(residual[foreground].mean()) * batch_size
                totals["target_mae"] += float(absolute[foreground].mean()) * batch_size
            if background.any():
                totals["background_mse"] += float(residual[background].mean()) * batch_size
                totals["background_mae"] += float(absolute[background].mean()) * batch_size
            predicted_mask = prediction.float() > float(mask_threshold)
            target_mask = target.float() > float(mask_threshold)
            metric_dims = tuple(range(1, predicted_mask.ndim))
            intersection = (predicted_mask & target_mask).sum(dim=metric_dims).float()
            predicted_count = predicted_mask.sum(dim=metric_dims).float()
            target_count = target_mask.sum(dim=metric_dims).float()
            union = predicted_count + target_count - intersection
            iou = torch.where(union > 0, intersection / union, torch.ones_like(union))
            dice_denominator = predicted_count + target_count
            dice = torch.where(
                dice_denominator > 0,
                2.0 * intersection / dice_denominator,
                torch.ones_like(dice_denominator),
            )
            precision = torch.where(
                predicted_count > 0,
                intersection / predicted_count,
                torch.where(target_count == 0, torch.ones_like(target_count), torch.zeros_like(target_count)),
            )
            recall = torch.where(
                target_count > 0,
                intersection / target_count,
                torch.ones_like(target_count),
            )
            totals["iou"] += float(iou.sum())
            totals["dice"] += float(dice.sum())
            totals["precision"] += float(precision.sum())
            totals["recall"] += float(recall.sum())
        sample_count += batch_size

    return {key: value / max(sample_count, 1) for key, value in totals.items()}


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": dict(metrics),
            # 保留旧字段名，兼容早期读取该 checkpoint 的脚本
            "val_loss": float(metrics.get("loss", float("nan"))),
        },
        path,
    )
