from __future__ import annotations

from collections import OrderedDict
from typing import Literal

import librosa
import numpy as np
import torch
import yaml

from music_source_separation.sdr import fast_evaluate


def parse_yaml(config_yaml: str) -> dict:
    r"""Parse yaml file."""
    
    with open(config_yaml, "r") as fr:
        return yaml.load(fr, Loader=yaml.FullLoader)


class LinearWarmUp:
    r"""Linear learning rate warm up scheduler.
    """
    def __init__(self, warm_up_steps: int) -> None:
        self.warm_up_steps = warm_up_steps

    def __call__(self, step: int) -> float:
        if step <= self.warm_up_steps:
            return step / self.warm_up_steps
        else:
            return 1.


@torch.no_grad()
def update_ema(ema: nn.Module, model: nn.Module, decay: float = 0.999) -> None:
    """Update EMA model weights and buffers from model."""

    # Parameters
    for e, m in zip(ema.parameters(), model.parameters()):
        e.mul_(decay).add_(m.data.float(), alpha=1 - decay)

    # Buffers (BN running stats, etc)
    for e, m in zip(ema.buffers(), model.buffers()):
        if m.dtype in [torch.bool, torch.long]:
            continue
        e.mul_(decay).add_(m.data.float(), alpha=1 - decay)


def requires_grad(model: nn.Module, flag=True) -> None:
    for p in model.parameters():
        p.requires_grad = flag


def calculate_sdr(
    output: np.ndarray, 
    target: np.ndarray, 
    sr: float, 
    mode: Literal["default", "fast"] = "default"
) -> float:
    r"""Calculate the SDR of separation result.

    Args:
        output: (c, l)
        target: (c, l)

    Returns:
        sdr: float
    """

    museval_sr = 44100
    output = librosa.resample(y=output, orig_sr=sr, target_sr=museval_sr)  # (c, l)
    target = librosa.resample(y=target, orig_sr=sr, target_sr=museval_sr)  # (c, l)

    if mode == "default":
        # Calculate SDR with official museval package
        import museval
        
        (sdrs, _, _, _) = museval.evaluate(
            references=target.T[None, :, :],  # shape: (sources_num, l, c)
            estimates=output.T[None, :, :]  # shape: (sources_num, l, c)
        )
    elif mode == "fast":
        # Calculate SDR to speed up by 10 times.
        sdrs = fast_evaluate(
            references=target,  # shape: (c, l)
            estimates=output  # shape: (c, l)
        )
    else:
        raise ValueError(mode)

    sdr = np.nanmedian(sdrs)

    return sdr