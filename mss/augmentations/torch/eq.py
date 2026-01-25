import random
from torch import Tensor
import numpy as np
import librosa
import torch
import torch.nn.functional as F

from einops import rearrange


class BatchRandomEQ: 
    def __init__(
        self, 
        min_db=-6.0,
        max_db=6.0,
        n_bands=8
    ):
        self.n_fft = 2048
        self.hop_length = 512

        self.min_db = min_db
        self.max_db = max_db
        self.n_bands = n_bands

    def __call__(self, x: Tensor) -> Tensor:
        r"""Batch EQ.

        b: batch_size
        c: channels_num
        l: audio_samples
        s: EQ bands

        Args:
            x: (b, c, l)

        Returns:
            x: (b, c, l)
        """

        B, C, L = x.shape
        device = x.device

        # STFT
        x = F.pad(x, (0, self.n_fft))
        x = stft(x, self.n_fft, self.hop_length)  # (b, c, t, f)

        # Random gain per band (b, s)
        gain_db = torch.empty(B, self.n_bands, device=device).uniform_(
            self.min_db, self.max_db)  # (b, s)

        gain = db_to_scale(gain_db)  # (b, s)

        n_freqs = x.shape[-1]
        gain = F.interpolate(gain[:, None, :], size=n_freqs, mode="linear", 
            align_corners=True).squeeze(1)  # (b, f)

        # Apply EQ
        x *= gain[:, None, None, :]  # (b, c, t, f)

        # ISTFT
        out = istft(x, self.n_fft, self.hop_length, L)  # (b, c, l)
        # print("eq:", gain)

        return out


def stft(x: Tensor, n_fft: int, hop_length: int) -> Tensor:
    r"""Compute STFT.

    Args:
        x: (b, c, l)

    Returns:
        out: (b, c, t, f)
    """
    B, C = x.shape[0 : 2]
    x = torch.stft(
        input=rearrange(x, 'b c l -> (b c) l'),  # (b*c, l)
        n_fft=n_fft,
        hop_length=hop_length,
        window=torch.hann_window(n_fft, device=x.device),
        normalized=True,
        return_complex=True
    )  # (b*c, f, t)
    out = rearrange(x, '(b c) f t -> b c t f', b=B, c=C)  # (b, c, t, f)
    return out


def istft(x: Tensor, n_fft: int, hop_length: int, length: int = None) -> Tensor:
    r"""Compute inverse STFT.

    Args:
        x: (b, c, t, f)

    Returns:
        out: (b, c, l)
    """
    B, C = x.shape[0 : 2]
    x = torch.istft(
        input=rearrange(x, 'b c t f -> (b c) f t'),  # (b*c, f, t)
        n_fft=n_fft,
        hop_length=hop_length,
        window=torch.hann_window(n_fft).to(x.device),
        normalized=True,
        length=length,
        return_complex=False,
    )  # (b*c, l)
    out = rearrange(x, '(b c) l -> b c l', b=B, c=C)  # (b, c, l)
    return out


def db_to_scale(db):
    scale = 10 ** (db / 20.)
    return scale