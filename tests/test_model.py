import torch
import yaml
from pathlib import Path
from torch import nn

from u3dgpr.model import U3DGPRNet


class _RecordIdentity(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = None

    def forward(self, x):
        self.seen = x.detach().clone()
        return x


def test_directional_slices_follow_txy_physical_axes():
    # Source storage is [t,x,y]; canonical model storage is [C,D,T]=[y,x,t].
    source_txy = torch.arange(5 * 2 * 3, dtype=torch.float32).reshape(5, 2, 3)
    canonical = source_txy.permute(2, 1, 0).unsqueeze(0).unsqueeze(0)

    vertical_model = U3DGPRNet(output_mode="vertical")
    vertical_recorder = _RecordIdentity()
    vertical_model.vertical_branch = vertical_recorder
    vertical_model(canonical)
    expected_vertical = canonical[:, 0].reshape(3, 1, 2, 5)
    assert torch.equal(vertical_recorder.seen, expected_vertical)
    # Every vertical plane fixes y/C and contains [x,t]=[D,T].
    assert torch.equal(vertical_recorder.seen[0, 0], source_txy[:, :, 0].permute(1, 0))

    crossed_model = U3DGPRNet(output_mode="channel_crossed")
    crossed_recorder = _RecordIdentity()
    crossed_model.channel_branch = crossed_recorder
    crossed_model(canonical)
    expected_crossed = canonical[:, 0].permute(0, 2, 1, 3).reshape(2, 1, 3, 5)
    assert torch.equal(crossed_recorder.seen, expected_crossed)
    # Every channel-crossed plane fixes x/D and contains [y,t]=[C,T].
    assert torch.equal(crossed_recorder.seen[0, 0], source_txy[:, 0, :].permute(1, 0))


def test_full_benchmark_compatible_shape_and_exact_initial_weights():
    model = U3DGPRNet().eval()
    x = torch.randn(1, 1, 16, 16, 32)
    with torch.inference_mode():
        output = model(x)
    assert output.reconstruction.shape == x.shape
    assert output.vertical.shape == x.shape
    assert output.channel_crossed.shape == x.shape
    assert output.fusion_weights.shape == (1, 2, 16, 16, 32)
    assert torch.equal(output.fusion_weights, torch.full_like(output.fusion_weights, 0.5))
    assert torch.allclose(
        output.raw_fused,
        0.5 * output.vertical + 0.5 * output.channel_crossed,
    )


def test_fusion_weights_are_independent_not_softmax_constrained():
    model = U3DGPRNet().eval()
    final_fusion = next(
        layer for layer in reversed(model.fusion.layers) if isinstance(layer, torch.nn.Conv3d)
    )
    with torch.no_grad():
        final_fusion.weight.zero_()
        final_fusion.bias.copy_(torch.tensor([2.0, -1.0]))
        output = model(torch.randn(1, 1, 16, 16, 32))
    assert torch.equal(output.fusion_weights[:, 0], torch.full_like(output.fusion_weights[:, 0], 2.0))
    assert torch.equal(output.fusion_weights[:, 1], torch.full_like(output.fusion_weights[:, 1], -1.0))


def test_full_mode_runs_refinement_after_raw_fusion():
    model = U3DGPRNet().eval()
    refinement_calls = []
    handle = model.refiner.register_forward_hook(
        lambda _module, _inputs, _output: refinement_calls.append(True)
    )
    with torch.inference_mode():
        output = model(torch.randn(1, 1, 16, 16, 32))
    handle.remove()
    assert refinement_calls == [True]
    assert output.raw_fused is not None
    assert output.reconstruction.shape == output.raw_fused.shape


def test_transposed_convolutions_recover_odd_size_directly():
    model = U3DGPRNet().eval()
    x = torch.randn(1, 1, 17, 17, 32)
    with torch.inference_mode():
        output = model(x)
    assert output.reconstruction.shape == x.shape


def test_output_heads_have_deterministic_background_initialization():
    model = U3DGPRNet(output_bias_init=4.0)
    for branch in (model.vertical_branch, model.channel_branch, model.refiner):
        assert torch.allclose(branch.output.weight, torch.full_like(branch.output.weight, 0.1))
        assert torch.allclose(branch.output.bias, torch.full_like(branch.output.bias, 4.0))


def test_random_initialization_preserves_input_dependence():
    torch.manual_seed(1)
    model = U3DGPRNet().eval()
    first = torch.randn(1, 1, 16, 16, 32)
    second = torch.randn(1, 1, 16, 16, 32)
    with torch.inference_mode():
        first_output = model(first)
        second_output = model(second)
    assert first_output.vertical.std() > 1e-6
    assert first_output.channel_crossed.std() > 1e-6
    assert first_output.reconstruction.std() > 1e-6
    assert not torch.allclose(first_output.reconstruction, second_output.reconstruction)


def test_backward_on_small_shape():
    model = U3DGPRNet()
    x = torch.randn(1, 1, 16, 16, 32)
    target = torch.randn_like(x)
    loss = torch.nn.functional.mse_loss(model(x).reconstruction, target)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_adaptation_removes_final_single_channel_normalization():
    model = U3DGPRNet(
        normalization="batch",
        normalize_final_decoder=False,
        channel_kernels=(3, 3, 3, 3, 3),
        channel_up_kernels=(2, 2, 2, 2),
    )
    for branch in (model.vertical_branch, model.channel_branch, model.refiner):
        final_layers = list(branch.decoders[-1].layers)
        assert not any(isinstance(layer, torch.nn.BatchNorm2d) for layer in final_layers[-2:])


def test_staged_forward_does_not_update_inactive_branch_statistics():
    model = U3DGPRNet(output_mode="vertical", normalization="batch").train()
    inactive_batch_norm = next(
        layer for layer in model.channel_branch.modules() if isinstance(layer, torch.nn.BatchNorm2d)
    )
    running_mean = inactive_batch_norm.running_mean.clone()
    output = model(torch.randn(1, 1, 16, 16, 32))
    assert output.channel_crossed is None
    assert output.fusion_weights is None
    assert torch.equal(inactive_batch_norm.running_mean, running_mean)


def test_training_config_keeps_both_branches_alive_at_benchmark_scale():
    config_path = Path(__file__).parents[1] / "configs" / "transfer_3dinvnet.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    model_kwargs = {
        key: value for key, value in config["model"].items() if key != "return_intermediates"
    }
    torch.manual_seed(int(config["seed"]))
    model = U3DGPRNet(**model_kwargs).train()
    x = 0.5 + 0.0027 * torch.randn(1, 1, 16, 16, 32)
    target = torch.full_like(x, 4.0)
    target[..., 6:10, 6:10, 12:20] = 18.0
    output = model(x)
    assert output.vertical.std() > 1e-5
    assert output.channel_crossed.std() > 1e-5

    torch.nn.functional.mse_loss(output.reconstruction, target).backward()
    vertical_gradient = sum(
        float(parameter.grad.square().sum())
        for parameter in model.vertical_branch.parameters()
        if parameter.grad is not None
    ) ** 0.5
    channel_gradient = sum(
        float(parameter.grad.square().sum())
        for parameter in model.channel_branch.parameters()
        if parameter.grad is not None
    ) ** 0.5
    assert vertical_gradient > 1e-4
    assert channel_gradient > 1e-4
