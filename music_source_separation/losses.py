import torch


def l1_loss(output: torch.Tensor, target: torch.Tensor) -> torch.float:
	r"""L1 loss."""
	return torch.mean(torch.abs(output - target))