from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def l1(output: Tensor, target: Tensor) -> torch.float:
	r"""L1 loss."""
	return torch.mean(torch.abs(output - target))