import math
import time

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from scipy.signal import firwin
from torch import Tensor

from mss.utils import fast_sdr

from .analytic import analytic_to_real, real_to_analytic
from .banks import erb_linear_banks, mel_linear_banks
from .convolve import polyphase_fftconvolve
from .upsample import polyphase_fftupsample
from .utils import shift_frequency


class SubbandFilter(nn.Module):
    r"""Save memory version. Split signal into subbands."""

    def __init__(
        self, 
        sr: int, 
        banks: list[tuple[int, int]],
        factor: int,
        chunk_size = 4,
        bandpass_filter_len = 48000,
        upsample_filter_len = 12000
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
        self.bandpass_filter_len = bandpass_filter_len
        self.upsample_filter_len = upsample_filter_len

        # Bandpass filter
        n_banks = len(banks)
        N = self.bandpass_filter_len - 1
        w = torch.zeros((n_banks, self.bandpass_filter_len))  # (k, n)
        for i in range(n_banks):
            if i == 0:
                w[i, 1 : ] = self.lowpass(banks[i][1], N)
            elif i == n_banks - 1:
                w[i, 1 :] = self.highpass(banks[i][0], N)
            else:
                w[i, 1 :] = self.bandpass(banks[i][0], banks[i][1], N)
        self.register_buffer("w", w)  # (k, n)

        # Check Nyquist sampling rate
        bandwidths = [bank[1] - bank[0] for bank in banks]  # (k,)
        assert max(bandwidths) <= sr / self.factor

        # Center frequency
        self.register_buffer("f_center", Tensor([np.mean(bank) for bank in banks]))  # (k,)

        # Upsample filter
        up = torch.zeros(self.upsample_filter_len)
        up[1 :] = torch.from_numpy(firwin(
            numtaps=self.upsample_filter_len - 1, 
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

        # self.factor = 10
        B, C, L = x.shape
        x = real_to_analytic(x)  # (b, c, l_up)

        x = bandpass_demodulate_downsample(
            x=rearrange(x, 'b c l -> (b c) l'),
            h=self.w, 
            sr=self.sr, 
            freq=-self.f_center, 
            factor=self.factor, 
            chunk_size=self.chunk_size
        )  # (b*c, k, l_down)
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
            sr=self.sr, 
            freq=self.f_center, 
            factor=self.factor, 
            chunk_size=self.chunk_size
        )
        x = rearrange(x, '(b c) l -> b c l', b=B)
        out = analytic_to_real(x)  # (b, c, l)

        return out

    def lowpass(self, f: float, n: int) -> Tensor:
        h = firwin(
            numtaps=n, 
            cutoff=f / (self.sr / 2), 
            pass_zero="lowpass",
            window=self.window_type
        )
        # from IPython import embed; embed(using=False); os._exit(0)
        # h = firwin(numtaps=5, cutoff=f / (self.sr / 2), pass_zero="lowpass",window=self.window_type)
        return torch.from_numpy(h)  # (n,)

    def bandpass(self, f1: float, f2: float, n: int) -> Tensor:
        h = firwin(
            numtaps=n, 
            cutoff=[f1 / (self.sr / 2), f2 / (self.sr / 2)], 
            pass_zero="bandpass",
            window=self.window_type
        )
        return torch.from_numpy(h)  # (n,)

    def highpass(self, f: float, n: int) -> Tensor:
        h = firwin(
            numtaps=n, 
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
        x: (b, l)
        h: (k, n)
        sr: int
        freq: (k,)
        factor: int

    Returns:
        out: (b, k, l_down)
    """
    B = x.shape[0]
    K = h.shape[0]
    L = math.ceil(x.shape[1] / factor)

    out = torch.empty(B, K, L, dtype=x.dtype, device=x.device)  # (b, k, l_out)
    k = 0

    while k < K:

        # Split into bands
        e = polyphase_fftconvolve(x, h[k : k + chunk_size, :], factor)  # (b, s, l_up)

        # Demodulate
        e = shift_frequency(e, freq[k : k + chunk_size], sr, factor)  # (b, s, l_up)
        
        # Downsample
        out[:, k : k + chunk_size, :] = e  # (b, s, l_down)

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
        e = x[:, k : k + chunk_size, :]
        e = rearrange(e, 'b s l -> (b s) l')  # (b*s, 1, l)
        e = polyphase_fftupsample(e * factor, up, factor)  # (b*s, l)
        e = rearrange(e, '(b s) l -> b s l', b=B) 
        
        # # Modulate
        e = shift_frequency(e, freq[k : k + chunk_size], sr)  # (b, s, l)

        # # Sum
        out.add_(e.sum(dim=1))  # (b, l)
        
        k += chunk_size

    return out


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
    
    for _ in range(2000):

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