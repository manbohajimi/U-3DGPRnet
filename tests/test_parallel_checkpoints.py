import torch

from train import load_branch_checkpoint
from u3dgpr.model import U3DGPRNet


def test_independent_directional_checkpoints_can_be_merged(tmp_path):
    vertical_source = U3DGPRNet()
    channel_source = U3DGPRNet()
    destination = U3DGPRNet()

    with torch.no_grad():
        for parameter in vertical_source.vertical_branch.parameters():
            parameter.fill_(0.25)
        for parameter in channel_source.channel_branch.parameters():
            parameter.fill_(-0.75)

    vertical_path = tmp_path / "vertical.pt"
    channel_path = tmp_path / "channel.pt"
    torch.save({"model": vertical_source.state_dict()}, vertical_path)
    torch.save({"model": channel_source.state_dict()}, channel_path)

    load_branch_checkpoint(destination, str(vertical_path), "vertical_branch")
    load_branch_checkpoint(destination, str(channel_path), "channel_branch")

    assert all(
        torch.equal(parameter, torch.full_like(parameter, 0.25))
        for parameter in destination.vertical_branch.parameters()
    )
    assert all(
        torch.equal(parameter, torch.full_like(parameter, -0.75))
        for parameter in destination.channel_branch.parameters()
    )
