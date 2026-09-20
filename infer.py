from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from u3dgpr.config import load_config
from u3dgpr.data import load_array, normalize_gpr, resize_volume
from u3dgpr.model import U3DGPRNet


def main() -> None:
    parser = argparse.ArgumentParser(description="Run U-3DGPR-Net inference")
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if config["device"] == "cuda" and torch.cuda.is_available() else "cpu")
    model = U3DGPRNet(**{k: v for k, v in config["model"].items() if k != "return_intermediates"}).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint.get("model", checkpoint))
    model.eval()

    data_config = config["data"]
    volume = load_array(args.input, key="gpr")
    axis_order = tuple(data_config.get("axis_order", (0, 1, 2)))
    if sorted(axis_order) != [0, 1, 2]:
        raise ValueError(f"axis_order must be a permutation of [0, 1, 2], got {axis_order}")
    volume = np.transpose(volume, axis_order)
    configured_shape = data_config.get("shape")
    if configured_shape is not None:
        volume = resize_volume(volume, tuple(configured_shape))
    else:
        volume = np.asarray(volume, dtype=np.float32)
    volume = normalize_gpr(
        volume,
        data_config["input_normalization"],
        input_min=float(data_config.get("input_min", -9.0)),
        input_max=float(data_config.get("input_max", 9.0)),
    )
    tensor = torch.from_numpy(volume)[None, None].to(device)
    with torch.inference_mode():
        result = model(tensor)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        permittivity=result.reconstruction[0, 0].cpu().numpy() * float(data_config["target_scale"]),
        vertical=result.vertical[0, 0].cpu().numpy(),
        channel_crossed=result.channel_crossed[0, 0].cpu().numpy(),
        fusion_weights=result.fusion_weights[0].cpu().numpy(),
    )


if __name__ == "__main__":
    main()
