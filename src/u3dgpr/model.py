"""Paper-faithful, shape-safe reconstruction of U-3DGPR-Net.

The paper specifies the branch topology and feature widths in Fig. 2 but does
not publish source code, padding, activation, or normalization details. This
implementation keeps all reported dimensions and makes ambiguous choices
explicit and easy to replace.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "leaky_relu":
        return nn.LeakyReLU(0.1, inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


def _normalization(name: str, channels: int) -> nn.Module:
    """Build an explicit signal-stabilization layer.

    The paper does not report normalization. ``batch`` is an engineering
    default used to make the very narrow Fig. 2 network trainable, not a
    claimed paper hyperparameter.
    """

    if name == "none":
        return nn.Identity()
    if name == "batch":
        return nn.BatchNorm2d(channels)
    if name == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    if name == "group":
        return nn.GroupNorm(1, channels)
    raise ValueError(f"Unsupported normalization: {name}")


class ConvBlock(nn.Module):
    """Three convolutions: one channel projection and two feature convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        activation: str,
        normalization: str,
        activate_last: bool = True,
        normalize_last: bool = True,
    ):
        super().__init__()
        padding = kernel_size // 2
        layers: list[nn.Module] = []
        for index in range(3):
            layers.append(
                nn.Conv2d(
                    in_channels if index == 0 else out_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                )
            )
            if index < 2 or normalize_last:
                layers.append(_normalization(normalization, out_channels))
            if index < 2 or activate_last:
                layers.append(_activation(activation))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class DirectionalUNet2D(nn.Module):
    """Small 2-D U-Net used for one acquisition direction.

    Transposed convolutions follow the paper. Interpolation only resolves odd
    dimensions (54 -> 27 -> 13 -> 6 -> 3) before concatenating skip features.
    """

    def __init__(
        self,
        features: Iterable[int] = (1, 2, 4, 8, 16),
        kernels: Iterable[int] = (3, 3, 3, 3, 3),
        activation: str = "relu",
        normalization: str = "batch",
        up_kernels: Iterable[int] = (2, 2, 2, 2),
        normalize_final_decoder: bool = True,
    ):
        super().__init__()
        widths = tuple(features)
        kernel_sizes = tuple(kernels)
        decoder_kernels = tuple(up_kernels)
        if len(widths) != 5 or len(kernel_sizes) != 5:
            raise ValueError("Fig. 2 uses five feature-extraction modules")
        if len(decoder_kernels) != 4:
            raise ValueError("Fig. 2 uses four decoding modules")

        self.encoders = nn.ModuleList()
        in_channels = 1
        for width, kernel in zip(widths, kernel_sizes):
            self.encoders.append(ConvBlock(in_channels, width, kernel, activation, normalization))
            in_channels = width
        self.pool = nn.MaxPool2d(2)

        reversed_widths = widths[::-1]
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        current = reversed_widths[0]
        for decoder_index, (kernel, skip_width) in enumerate(
            zip(decoder_kernels, reversed_widths[1:])
        ):
            self.upconvs.append(
                nn.ConvTranspose2d(current, skip_width, kernel_size=kernel, stride=2)
            )
            self.decoders.append(
                ConvBlock(
                    skip_width * 2,
                    skip_width,
                    3,
                    activation,
                    normalization,
                    # The last feature map must be signed. A ReLU here makes
                    # the scalar output head one-sided and caused the observed
                    # seed-dependent collapse when its sole weight was negative.
                    activate_last=decoder_index != len(decoder_kernels) - 1,
                    # A one-channel BatchNorm immediately before the scalar
                    # output head became numerically ill-conditioned on the
                    # sparse 3DInvNet adaptation (running_var ~= 1e-3). Keep
                    # legacy behavior configurable for old checkpoints, but
                    # allow the final decoder feature to remain unnormalized.
                    normalize_last=(
                        decoder_index != len(decoder_kernels) - 1
                        or normalize_final_decoder
                    ),
                )
            )
            current = skip_width
        self.output = nn.Conv2d(widths[0], 1, kernel_size=1)

    @staticmethod
    def _match(x: Tensor, reference: Tensor) -> Tensor:
        if x.shape[-2:] != reference.shape[-2:]:
            x = F.interpolate(x, size=reference.shape[-2:], mode="bilinear", align_corners=False)
        return x

    def forward(self, x: Tensor) -> Tensor:
        skips: list[Tensor] = []
        for index, encoder in enumerate(self.encoders):
            x = encoder(x)
            skips.append(x)
            if index != len(self.encoders) - 1:
                x = self.pool(x)

        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips[:-1])):
            x = self._match(upconv(x), skip)
            x = decoder(torch.cat((skip, x), dim=1))
        return self.output(x)


class MultiAngleFusion(nn.Module):
    """Six 1x1x1 convolutions learn voxel-wise weights for A and B (Eq. 5)."""

    def __init__(self, hidden_channels: int = 8, activation: str = "relu"):
        super().__init__()
        layers: list[nn.Module] = []
        channels = [2, hidden_channels, hidden_channels, hidden_channels, hidden_channels, hidden_channels, 2]
        for index, (in_channels, out_channels) in enumerate(zip(channels, channels[1:])):
            conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
            layers.append(conv)
            if index < 5:
                layers.append(_activation(activation))
        self.layers = nn.Sequential(*layers)

    def forward(self, vertical: Tensor, channel_crossed: Tensor) -> tuple[Tensor, Tensor]:
        logits = self.layers(torch.cat((vertical, channel_crossed), dim=1))
        weights = torch.softmax(logits, dim=1)
        raw_fused = weights[:, 0:1] * vertical + weights[:, 1:2] * channel_crossed
        return raw_fused, weights


@dataclass(frozen=True)
class U3DGPROutput:
    reconstruction: Tensor
    vertical: Tensor | None
    channel_crossed: Tensor | None
    fusion_weights: Tensor | None
    raw_fused: Tensor | None


class U3DGPRNet(nn.Module):
    """Dual-direction U-3DGPR-Net for volumes ordered as [B, 1, C, X, T]."""

    def __init__(
        self,
        features: Iterable[int] = (1, 2, 4, 8, 16),
        activation: str = "relu",
        normalization: str = "batch",
        fusion_hidden_channels: int = 8,
        output_mode: str = "full",
        output_bias_init: float = 0.0,
        output_weight_init: float = 0.1,
        fusion_logit_init_std: float = 1.0e-3,
        channel_kernels: Iterable[int] = (3, 3, 3, 1, 1),
        channel_up_kernels: Iterable[int] = (1, 1, 2, 2),
        normalize_final_decoder: bool = True,
    ):
        super().__init__()
        valid_modes = {"full", "vertical", "channel_crossed", "equal_fusion", "learned_fusion"}
        if output_mode not in valid_modes:
            raise ValueError(f"output_mode must be one of {sorted(valid_modes)}, got {output_mode!r}")
        self.output_mode = output_mode
        features = tuple(features)
        self.vertical_branch = DirectionalUNet2D(
            features=features,
            kernels=(3, 3, 3, 3, 3),
            activation=activation,
            normalization=normalization,
            up_kernels=(2, 2, 2, 2),
            normalize_final_decoder=normalize_final_decoder,
        )
        # Section 2.2: the deepest channel-crossed modules use 1x1 kernels;
        # its first decoder upsampling also uses a 1x1 transposed convolution.
        self.channel_branch = DirectionalUNet2D(
            features=features,
            kernels=channel_kernels,
            activation=activation,
            normalization=normalization,
            up_kernels=channel_up_kernels,
            normalize_final_decoder=normalize_final_decoder,
        )
        self.fusion = MultiAngleFusion(fusion_hidden_channels, activation)
        self.refiner = DirectionalUNet2D(
            features=features,
            kernels=(3, 3, 3, 3, 3),
            activation=activation,
            normalization=normalization,
            up_kernels=(2, 2, 2, 2),
            normalize_final_decoder=normalize_final_decoder,
        )
        self._reset_parameters(activation)

        # Deterministic signed heads remove the single-weight sign lottery.
        # The 3DInvNet adaptation overrides the generic zero bias with its
        # known relative-permittivity background value of 4.
        for branch in (self.vertical_branch, self.channel_branch, self.refiner):
            nn.init.constant_(branch.output.weight, float(output_weight_init))
            if branch.output.bias is not None:
                nn.init.constant_(branch.output.bias, float(output_bias_init))

        # Near-zero logits preserve the paper's approximately equal initial
        # contribution without the exact-zero gradient barrier of zero weights.
        final_fusion = next(
            layer for layer in reversed(self.fusion.layers) if isinstance(layer, nn.Conv3d)
        )
        nn.init.normal_(final_fusion.weight, mean=0.0, std=float(fusion_logit_init_std))
        nn.init.zeros_(final_fusion.bias)

    def _reset_parameters(self, activation: str) -> None:
        """Use activation-aware initialization so input variance survives."""

        nonlinearity = "leaky_relu" if activation == "leaky_relu" else "relu"
        negative_slope = 0.1 if activation == "leaky_relu" else 0.0
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(
                    module.weight,
                    a=negative_slope,
                    mode="fan_in",
                    nonlinearity=nonlinearity,
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def _validate(x: Tensor) -> None:
        if x.ndim != 5 or x.shape[1] != 1:
            raise ValueError(f"Expected [B, 1, channels, survey, time], got {tuple(x.shape)}")

    def forward(self, x: Tensor) -> U3DGPROutput:
        self._validate(x)
        batch, _, channels, survey, time = x.shape

        vertical: Tensor | None = None
        channel_crossed: Tensor | None = None
        if self.output_mode != "channel_crossed":
            vertical_slices = x[:, 0].reshape(batch * channels, 1, survey, time)
            vertical = self.vertical_branch(vertical_slices)
            vertical = vertical.reshape(batch, channels, survey, time).unsqueeze(1)
            if self.output_mode == "vertical":
                # Staged training must not execute inactive branches: doing so
                # used to update their BatchNorm running statistics even though
                # the optimizer did not own their parameters.
                return U3DGPROutput(vertical, vertical, None, None, None)

        crossed_slices = x[:, 0].permute(0, 2, 1, 3).reshape(batch * survey, 1, channels, time)
        channel_crossed = self.channel_branch(crossed_slices)
        channel_crossed = channel_crossed.reshape(batch, survey, channels, time)
        channel_crossed = channel_crossed.permute(0, 2, 1, 3).unsqueeze(1)
        if self.output_mode == "channel_crossed":
            return U3DGPROutput(channel_crossed, None, channel_crossed, None, None)

        assert vertical is not None
        raw_fused, weights = self.fusion(vertical, channel_crossed)
        if self.output_mode == "equal_fusion":
            reconstruction = 0.5 * vertical + 0.5 * channel_crossed
        elif self.output_mode == "learned_fusion":
            reconstruction = raw_fused
        else:
            refine_slices = raw_fused[:, 0].reshape(batch * channels, 1, survey, time)
            reconstruction = self.refiner(refine_slices)
            reconstruction = reconstruction.reshape(batch, channels, survey, time).unsqueeze(1)
        return U3DGPROutput(reconstruction, vertical, channel_crossed, weights, raw_fused)
