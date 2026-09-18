import torch

from u3dgpr.model import U3DGPRNet


def test_full_paper_shape_and_near_equal_initial_weights():
    model = U3DGPRNet().eval()
    x = torch.randn(1, 1, 20, 54, 512)
    with torch.inference_mode():
        output = model(x)
    assert output.reconstruction.shape == x.shape
    assert output.vertical.shape == x.shape
    assert output.channel_crossed.shape == x.shape
    assert output.fusion_weights.shape == (1, 2, 20, 54, 512)
    assert torch.allclose(output.fusion_weights.sum(dim=1), torch.ones_like(output.fusion_weights[:, 0]))
    assert (output.fusion_weights - 0.5).abs().max() < 0.05


def test_output_heads_have_deterministic_background_initialization():
    model = U3DGPRNet(output_bias_init=4.0)
    for branch in (model.vertical_branch, model.channel_branch, model.refiner):
        assert torch.allclose(branch.output.weight, torch.full_like(branch.output.weight, 0.1))
        assert torch.allclose(branch.output.bias, torch.full_like(branch.output.bias, 4.0))


def test_random_initialization_preserves_input_dependence():
    torch.manual_seed(1)
    model = U3DGPRNet().eval()
    first = torch.randn(1, 1, 20, 54, 512)
    second = torch.randn(1, 1, 20, 54, 512)
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
    model = U3DGPRNet(output_mode="vertical").train()
    inactive_batch_norm = next(
        layer for layer in model.channel_branch.modules() if isinstance(layer, torch.nn.BatchNorm2d)
    )
    running_mean = inactive_batch_norm.running_mean.clone()
    output = model(torch.randn(1, 1, 16, 16, 32))
    assert output.channel_crossed is None
    assert output.fusion_weights is None
    assert torch.equal(inactive_batch_norm.running_mean, running_mean)
