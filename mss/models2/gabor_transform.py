from __future__ import annotations

import math

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from mss.models2.fractional_stft import fractional_istft, fractional_stft


class GaborTransform(nn.Module):
    def __init__(self, n_ffts: list, hop_lengths: list, oversampling_factors: list):
        super().__init__()

        self.n_ffts = n_ffts
        self.hop_lengths = hop_lengths
        self.oversampling_factors = oversampling_factors
        self.n_windows = len(self.n_ffts)

        for n_fft in self.n_ffts:
            self.register_buffer(f"window_{n_fft}", torch.hann_window(n_fft))

    def transform(self, x: Tensor) -> Tensor:
        r"""Encode audio into Gabor features.

        w: n_windows
        b: batch_size
        c: n_channels
        L: audio_samples
        f: freq_bins
        r: fractions
        F: fractional freq bins = f*r

        Args:
            x: (b, c, l)

        Returns:
            out (w, b, c, t, F)
        """
        
        out = []  # (w, b, c, t, F)
        B = x.shape[0]
        x = rearrange(x, 'b c l -> (b c) l')  # (b*c, L)

        for i in range(self.n_windows):
            n_fft = self.n_ffts[i]
            hop_length = self.hop_lengths[i]
            r = self.oversampling_factors[i]
            window = getattr(self, f"window_{n_fft}")
            y = fractional_stft(x, n_fft, hop_length, r, window) / math.sqrt(self.n_windows)  # (b*c, t, F)
            y = rearrange(y, '(b c) t f -> b c t f', b=B)  # (b, c, t, F)
            out.append(y)

        return out

    def inverse_transform(self, x: list[Tensor], length: int | None) -> Tensor:
        r"""Decode Gabor features into audio.

        w: n_windows
        b: batch_size
        c: n_channels
        L: audio_samples
        n: window_size
        f: freq_bins
        r: fractions
        F: fractional freq bins = f*r

        Args:
            x: (w, b, c, t, F)
            length: int

        Returns:
            out: (b, c, L)
        """
        
        B, C = x[0].shape[0 : 2]
        out = torch.zeros((B, C, length), device=x[0].device)  # (b, c, L)
        
        for i in range(self.n_windows):
            n_fft = self.n_ffts[i]
            hop_length = self.hop_lengths[i]
            r = self.oversampling_factors[i]
            window = getattr(self, f"window_{n_fft}")  # (n,)
            y = rearrange(x[i], 'b c t f -> (b c) t f')  # (b*c, t, F)
            y = fractional_istft(y, n_fft, hop_length, r, window, length) / math.sqrt(self.n_windows)  # (b*c, L)
            y = rearrange(y, '(b c) l -> b c l', b=B)  # (b, c, L)
            out.add_(y)  # (b, c, L)
        
        return out



if __name__ == '__main__':

    sr = 48000
    device = "cuda"
    
    x = torch.randn((4, 2, sr * 2), device=device)

    gabor = GaborTransform(
        n_ffts=[512, 2048, 8192],
        hop_lengths=[128, 512, 2048],
        oversampling_factors=[16, 16, 16],
    ).to(device)

    y = gabor.transform(x)
    x_hat = gabor.inverse_transform(y, x.shape[-1])
    print("Error: {}".format((x - x_hat).abs().mean()))