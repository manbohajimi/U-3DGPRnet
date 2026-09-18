"""Dependency-free functional smoke test (no pytest required)."""

from __future__ import annotations

import time

import torch

from u3dgpr.model import U3DGPRNet


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = U3DGPRNet().to(device).eval()
    volume = torch.randn(1, 1, 20, 54, 512, device=device)
    second_volume = torch.randn_like(volume)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(volume)
        second_output = model(second_volume)
    if device.type == "cuda":
        torch.cuda.synchronize()
    assert output.reconstruction.shape == volume.shape
    assert torch.allclose(
        output.fusion_weights.sum(dim=1), torch.ones_like(output.fusion_weights[:, 0])
    )
    assert (output.fusion_weights - 0.5).abs().max() < 0.05
    assert output.vertical.std() > 1e-4
    assert output.channel_crossed.std() > 1e-4
    assert output.reconstruction.std() > 1e-4
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

    backward_model = U3DGPRNet().to(device)
    small = torch.randn(1, 1, 16, 16, 32, device=device)
    loss = backward_model(small).reconstruction.square().mean()
    loss.backward()
    assert any(parameter.grad is not None for parameter in backward_model.parameters())
    print("backward=ok")


if __name__ == "__main__":
    main()
