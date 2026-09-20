"""Dependency-free functional smoke test (no pytest required)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import yaml

from u3dgpr.model import U3DGPRNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Config-aware U-3DGPR-Net smoke test")
    parser.add_argument("--config", default="configs/transfer_3dinvnet.yaml")
    return parser.parse_args()


def synthetic_input(shape: tuple[int, int, int], normalization: str, device: torch.device) -> torch.Tensor:
    """Use the scale produced by the configured preprocessing, not N(0, 1) blindly."""

    if normalization == "fixed_minmax":
        # Measured 3DInvNet volumes are centered near 0 before (x+9)/18;
        # after scaling they are near 0.5 with std around 0.0027.
        return 0.5 + 0.0027 * torch.randn(1, 1, *shape, device=device)
    if normalization == "zscore":
        return torch.randn(1, 1, *shape, device=device)
    # Raw 3DInvNet sample amplitudes are O(1e-2), not unit Gaussian.
    return 0.027 * torch.randn(1, 1, *shape, device=device)


def gradient_norm(module: torch.nn.Module) -> float:
    squared = sum(
        float(parameter.grad.detach().float().square().sum())
        for parameter in module.parameters()
        if parameter.grad is not None
    )
    return squared**0.5


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    torch.manual_seed(int(config.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_kwargs = {
        key: value
        for key, value in config["model"].items()
        if key != "return_intermediates"
    }
    model = U3DGPRNet(**model_kwargs).to(device).eval()
    # CUDA checks the actual 3DInvNet benchmark volume; the smaller CPU shape
    # keeps this dependency-free smoke test practical on developer machines.
    spatial_shape = (128, 128, 128) if device.type == "cuda" else (16, 16, 32)
    input_normalization = str(config["data"]["input_normalization"])
    volume = synthetic_input(spatial_shape, input_normalization, device)
    second_volume = synthetic_input(spatial_shape, input_normalization, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(volume)
        second_output = model(second_volume)
    if device.type == "cuda":
        torch.cuda.synchronize()
    assert output.reconstruction.shape == volume.shape
    assert torch.equal(output.fusion_weights, torch.full_like(output.fusion_weights, 0.5))
    assert output.vertical.std() > 1e-5, "vertical branch collapsed on configured input scale"
    assert output.channel_crossed.std() > 1e-5, "channel branch collapsed on configured input scale"
    assert output.reconstruction.std() > 1e-5, "full output collapsed on configured input scale"
    assert not torch.allclose(output.reconstruction, second_output.reconstruction)
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else float("nan")
    print(f"device={device} shape={tuple(output.reconstruction.shape)}")
    print(f"parameters={sum(parameter.numel() for parameter in model.parameters()):,}")
    print(
        "signal_std="
        f"vertical:{float(output.vertical.std()):.6f},"
        f"crossed:{float(output.channel_crossed.std()):.6f},"
        f"full:{float(output.reconstruction.std()):.6f}"
    )
    print(f"forward_seconds={elapsed:.4f} peak_memory_mb={peak:.1f}")

    backward_model = U3DGPRNet(**model_kwargs).to(device).train()
    small = synthetic_input((16, 16, 32), input_normalization, device)
    target = torch.full_like(small, 4.0)
    target[..., 6:10, 6:10, 12:20] = 18.0
    loss = torch.nn.functional.mse_loss(backward_model(small).reconstruction, target)
    loss.backward()
    vertical_gradient = gradient_norm(backward_model.vertical_branch)
    channel_gradient = gradient_norm(backward_model.channel_branch)
    assert vertical_gradient > 1e-4, "vertical branch gradient is effectively dead"
    assert channel_gradient > 1e-4, "channel branch gradient is effectively dead"
    print(
        f"backward=ok vertical_grad={vertical_gradient:.6e} "
        f"channel_grad={channel_gradient:.6e}"
    )


if __name__ == "__main__":
    main()
