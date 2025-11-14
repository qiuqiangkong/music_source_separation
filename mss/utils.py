from __future__ import annotations

from collections import OrderedDict
from typing import Literal

import librosa
import numpy as np
import torch
import yaml
import math
import museval

from einops import rearrange
from scipy.signal import get_window


def parse_yaml(config_yaml: str) -> dict:
    r"""Parse yaml file."""
    
    with open(config_yaml, "r") as fr:
        return yaml.load(fr, Loader=yaml.FullLoader)


@torch.no_grad()
def update_ema(ema: nn.Module, model: nn.Module, decay: float = 0.999) -> None:
    r"""Update EMA model weights and buffers from model."""

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


def separate_overlap_add(
    model: nn.Module, 
    audio: Tensor, 
    segment_samples: int, 
    hop_length: int,
    batch_size: int
):
    r"""Split audio into clips. Separate each clip. Concatenate the results.

    b: batch_size
    c: channels_num
    L: audio_samples
    l: segment_samples
    n: segments_num

    Args:
        model (nn.Module)
        audio (np.ndarray): (c, L)
        segment_samples (int)
        hop_length (int)
        batch_size (int)

    Returns:
        output: (c, L)
    """

    device = next(model.parameters()).device
    
    audio_samples = audio.shape[1]
    
    if audio_samples < segment_samples:
        full_samples = segment_samples
    else:
        full_samples = segment_samples + math.ceil((audio_samples - segment_samples) / hop_length) * hop_length
    
    audio = librosa.util.fix_length(data=audio, size=full_samples, axis=-1)  # (c, n*l)

    window = get_window(window="hamming", Nx=segment_samples)
    
    segments = librosa.util.frame(
        audio, 
        frame_length=segment_samples, 
        hop_length=hop_length
    )  # (c, l, n)

    segments = rearrange(segments, 'c l n -> n c l')  # (n, c, l)
    clips_num = segments.shape[0]

    p = 0
    outputs = []

    while p < clips_num:

        x = torch.Tensor(segments[p : p + batch_size].copy()).to(device)  # (b, c, t)

        with torch.no_grad():
            model.eval()
            out = model(x)  # (b, c, l)

        outputs.append(out.cpu().numpy())
        p += batch_size

    outputs = np.concatenate(outputs, axis=0)  # (n, c, l)

    y = np.zeros_like(audio)
    ola = np.zeros_like(audio)

    for i in range(clips_num):
        y[:, i * hop_length : i * hop_length + segment_samples] += outputs[i] * window
        ola[:, i * hop_length : i * hop_length + segment_samples] += window

    y = y / ola
    y = y[:, 0 : audio_samples]

    return y


def calculate_sdr(
    output: np.ndarray, 
    target: np.ndarray, 
    sr: float, 
    fast_only: bool = False
) -> tuple[float, float]:
    r"""Compute the SDR of separation result.

    c: channels_num
    L: audio_samples
    n: segments_num

    Args:
        output (np.ndarray): (c, L)
        target (np.ndarray): (c, L)

    Returns:
        sdr (float)
        fast_sdr (float)
    """

    museval_sr = 44100
    output = librosa.resample(y=output, orig_sr=sr, target_sr=museval_sr)  # (c, l)
    target = librosa.resample(y=target, orig_sr=sr, target_sr=museval_sr)  # (c, l)

    if fast_only:
        sdr = None
    else:
        # Compute SDR with official museval package
        (sdrs, _, _, _) = museval.evaluate(
            references=target.T[None, :, :],  # (sources_num, L, c)
            estimates=output.T[None, :, :]  # (sources_num, L, c)
        )  # sdrs: (sources_num, n)
        sdr = np.nanmedian(sdrs)

    # Fast SDR
    fast_sdrs = fast_evaluate(
        references=target,  # (c, L)
        estimates=output  # (c, L)
    )
    fast_sdr = np.nanmedian(fast_sdrs)
    
    return sdr, fast_sdr


def fast_evaluate(
    references: np.ndarray, 
    estimates: np.ndarray, 
    win: int =1 * 44100, 
    hop: int =1 * 44100
):
    r"""Fast version to compute SDR of separation result. This function is 
    200 times faster than museval.evaluate(). The computed SDR is sometimes 
    lower than museval.evaluate()

    c: channels_num
    L: audio_samples
    l: segment_samples
    n: segments_num

    Args:
        output (np.ndarray): (c, L)
        target (np.ndarray): (c, L)

    Returns:
        sdr (float): (n,)
    """

    refs = librosa.util.frame(references, frame_length=win, hop_length=hop)  # (c, l, n)
    ests = librosa.util.frame(estimates, frame_length=win, hop_length=hop)  # (c, l, n)

    segs_num = refs.shape[2]
    sdrs = []

    for n in range(segs_num):
        sdr = fast_sdr(ref=refs[:, :, n], est=ests[:, :, n])
        sdrs.append(sdr)
    
    return np.stack(sdrs)


def fast_sdr(
    ref: np.ndarray, 
    est: np.ndarray, 
    eps: float = 1e-10
):
    r"""Compute SDR."""
    noise = est - ref
    numerator = np.clip(a=np.mean(ref ** 2), a_min=eps, a_max=None)
    denominator = np.clip(a=np.mean(noise ** 2), a_min=eps, a_max=None)
    sdr = 10. * np.log10(numerator / denominator)
    return sdr