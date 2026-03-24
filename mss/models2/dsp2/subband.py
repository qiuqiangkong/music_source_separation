import math
import time

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from scipy.signal import firwin
from torch import Tensor

from mss.models2.dsp2.analytic import analytic_to_real, real_to_analytic
from mss.models2.dsp2.banks import mel_linear_banks, erb_linear_banks
from mss.models2.dsp2.convolve import fftconvolve_complex
from mss.utils import fast_sdr


class SubbandFilter(nn.Module):
    r"""Save memory version. Split signal into subbands."""

    def __init__(
        self, 
        sr: int, 
        banks: list[tuple[int, int]],
        factor: int,
        chunk_size = 4,
    ):
        r"""
        k: n_bands
        m: filter_len
        """

        super().__init__()

        self.sr = sr
        self.banks = banks
        self.window_type = "hamming"
        self.factor = factor
        self.chunk_size = chunk_size
        self.bandpass_filter_len = 48001
        self.upsample_filter_len = 10001

        assert self.bandpass_filter_len % 2 == 1
        assert self.upsample_filter_len % 2 == 1

        # Bandpass filter
        n_banks = len(banks)
        w = torch.empty((n_banks, self.bandpass_filter_len))  # (k, m)
        for i in range(n_banks):
            if i == 0:
                w[i] = self.lowpass(banks[i][1])
            elif i == n_banks - 1:
                w[i] = self.highpass(banks[i][0])
            else:
                w[i] = self.bandpass(banks[i][0], banks[i][1])
        self.register_buffer("w", w[:, None, :])  # (m, 1, kernel_size)

        # Check Nyquist sampling rate
        bandwidths = [bank[1] - bank[0] for bank in banks]  # (k,)
        assert max(bandwidths) <= sr / self.factor

        # Center frequency
        self.register_buffer("f_center", Tensor([np.mean(bank) for bank in banks]))  # (k,)

        # Upsample filter
        up = torch.from_numpy(firwin(
            numtaps=self.upsample_filter_len, 
            cutoff=1. / self.factor, 
            pass_zero="lowpass",
            window="hamming"
        ))
        self.register_buffer("up", up)

    def analysis(self, x: Tensor) -> Tensor:
        r"""Split signal into subbands.

        b: batch_size
        c: audio_channels
        l: audio_samples
        k: n_bands

        Args:
            x: (b, c, l)

        Returns:
            out: (b, c, k, l)
        """ 

        B = x.shape[0]
        x = real_to_analytic(x)  # (b, c, l)
        
        x = bandpass_demodulate_downsample(
            x=rearrange(x, 'b c l -> (b c) 1 l'),  # (b*c, 1, l)
            h=self.w, 
            sr=sr, 
            freq=-self.f_center, 
            factor=self.factor, 
            chunk_size=self.chunk_size
        )  # (b*c, k, l_out)
        out = rearrange(x, '(b c) k l -> b c k l', b=B)

        return out
    

    def synthesis(self, x: Tensor) -> Tensor:
        r"""Sum subband signals into original signal.

        b: batch_size
        c: audio_channels
        l: audio_samples
        k: n_bands

        Args:
            x: (b, c, k, l)

        Returns:
            out: (b, c, l)
        """
        B = x.shape[0]

        x = upsample_modulate_sum(
            x=rearrange(x, 'b c k l -> (b c) k l'), 
            up=self.up, 
            sr=sr, 
            freq=self.f_center, 
            factor=self.factor, 
            chunk_size=self.chunk_size
        )
        x = rearrange(x, '(b c) l -> b c l', b=B)
        out = analytic_to_real(x)  # (b, c, l)

        return out


    def lowpass(self, f: float) -> Tensor:
        h = firwin(
            numtaps=self.bandpass_filter_len, 
            cutoff=f / (self.sr / 2), 
            pass_zero="lowpass",
            window=self.window_type
        )
        return torch.from_numpy(h)  # (n,)

    def bandpass(self, f1: float, f2: float) -> Tensor:
        h = firwin(
            numtaps=self.bandpass_filter_len, 
            cutoff=[f1 / (self.sr / 2), f2 / (self.sr / 2)], 
            pass_zero="bandpass",
            window=self.window_type
        )
        return torch.from_numpy(h)  # (n,)

    def highpass(self, f: float) -> Tensor:
        h = firwin(
            numtaps=self.bandpass_filter_len, 
            cutoff=f / (self.sr / 2), 
            pass_zero="highpass",
            window=self.window_type
        )
        return torch.from_numpy(h)  # (n,)


def bandpass_demodulate_downsample(
    x: Tensor, 
    h: Tensor, 
    sr: int, 
    freq: Tensor, 
    factor: int,
    chunk_size: int
) -> Tensor:
    r"""
    1. Bandpass: Split signal into subbands
    2. Demodulate: Shift the spectrum to baseband ω=0
    3. Downsample.

    b: batch_size
    l: audio_samples
    k: n_bands
    n: filter_len

    Args:
        x: (b, 1, l)
        h: (k, 1, n)
        sr: int
        freq: (k,)
        factor: int

    Returns:
        out: (b, k, l_down)
    """
    B = x.shape[0]
    K = h.shape[0]
    L = math.ceil(x.shape[2] / factor)

    out = torch.empty(B, K, L, dtype=x.dtype, device=x.device)  # (b, k, l_out)
    k = 0

    while k < K:

        # Split into bands
        e = fftconvolve_complex(x, h[k : k + chunk_size, :, :])  # (b, s, l_in)
        
        # Demodulate
        e = shift_frequency(e, freq[k : k + chunk_size], sr)  # (b, s, l_in)

        # Downsample
        out[:, k : k + chunk_size, :] = e[:, :, 0 :: factor]  # (b, 1, l_out)

        k += chunk_size

    return out


def upsample_modulate_sum(
    x: Tensor, 
    up: Tensor, 
    sr: int, 
    freq: Tensor, 
    factor: int,
    chunk_size: int
):
    r"""
    1. Upsample: Use sinc function to upsample a signal
    2. Modulate: Shift the baseband signal to ω0
    3. Sum: Sum subband signals

    b: batch_size
    l: audio_samples
    k: n_bands
    n: filter_len

    Args:
        x: (b, k, l_down)
        up: (n,)

    Returns:
        out: (b, l)
    """

    B, K = x.shape[0 : 2]
    L = x.shape[2] * factor
    
    out = torch.zeros(B, L, dtype=x.dtype, device=x.device)  # (b, l)
    k = 0

    while k < K:
        
        # Upsample
        e = torch.zeros(B, min(chunk_size, K - k), L, dtype=x.dtype, device=x.device)
        e[:, :, ::factor] = x[:, k : k + chunk_size, :]  # (b, s, l)
        e = rearrange(e, 'b s l -> (b s) 1 l')  # (b*s, 1, l)
        e = fftconvolve_complex(e * factor, up[None, None, :])  # (b*s, 1, l)
        e = rearrange(e, '(b s) 1 l -> b s l', b=B)  # (b, s, l)

        # # Modulate
        e = shift_frequency(e, freq[k : k + chunk_size], sr)  # (b, s, l)

        # # Sum
        out.add_(e.sum(dim=1))  # (b, l)

        k += chunk_size

    return out


def shift_frequency(x: Tensor, f: Tensor, sr: int, n=1) -> Tensor:
    r"""Shift frequency to -ω0.

    X(j(ω-ω0)) = e^{j(ω0)n}X(jω)

    k: n_bands
    l: audio_samples

    Args:
        x: (any, k, l)
        f: (k,)

    Returns:
        out: (any, k, l)
    """
    omega = f / (sr / 2) * math.pi  # (k,)
    t = torch.arange(x.shape[-1], device=x.device) * n  # (l,)
    a = torch.exp(1.j * omega[:, None] * t[None, :])
    x.mul_(a)
    return x


if __name__ == '__main__':
    
    sr = 48000
    n_bands = 256
    max_bandwidth = 800
    chunk_size = 16  # Try to tune this to balance RAM and computation speed
    factor = sr // max_bandwidth
    device = "cuda"

    # Melbanks
    # banks = mel_linear_banks(sr=sr, n_bands=n_bands, max_bandwidth=max_bandwidth)
    banks = erb_linear_banks(sr=sr, n_bands=n_bands, max_bandwidth=max_bandwidth)
    sb_filter = SubbandFilter(sr, banks, factor, chunk_size=chunk_size).to(device)

    for _ in range(20):

        # Audio
        rs = np.random.RandomState(1234)
        audio = rs.uniform(low=-1, high=1, size=(4, 2, sr * 2))
        audio = Tensor(audio).to(device)  # (c, l)
                
        # Analysis
        t0 = time.time()
        x = sb_filter.analysis(audio)  # (b, c, k, l)
        y = sb_filter.synthesis(x)
        
        # Print
        t1 = time.time() - t0
        sdr = fast_sdr(audio.cpu().numpy(), y.cpu().numpy())
        print(f"time: {t1:.4f} s, latent: {x.shape}, SDR: {sdr:.2f} dB")