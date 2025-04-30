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
def update_ema(ema_model: nn.Module, model: nn.Module, decay=0.999) -> None:

    # Moving average of parameters
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

    # Moving average of buffers. Patch for BN, etc
    ema_buffers = OrderedDict(ema_model.named_buffers())
    model_buffers = OrderedDict(model.named_buffers())

    for name, buffer in model_buffers.items():
        if buffer.dtype in [torch.long]:
            continue
        ema_buffers[name].mul_(decay).add_(buffer.data, alpha=1 - decay)


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

    import museval

    museval_sr = 44100
    output = librosa.resample(y=output, orig_sr=sr, target_sr=museval_sr)  # (c, l)
    target = librosa.resample(y=target, orig_sr=sr, target_sr=museval_sr)  # (c, l)

    if mode == "default":
        # Calculate SDR with official museval package
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