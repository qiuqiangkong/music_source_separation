import math
import time

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from scipy.signal import firwin
from torch import Tensor
import torch.nn.functional as F

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
        self.bandpass_filter_len = 48000
        self.upsample_filter_len = 12000

        # Bandpass filter
        n_banks = len(banks)
        N = self.bandpass_filter_len
        w = torch.zeros((n_banks, N))  # (k, m)
        for i in range(n_banks):
            if i == 0:
                w[i, 1 : ] = self.lowpass(banks[i][1], N - 1)
            elif i == n_banks - 1:
                w[i, 1 :] = self.highpass(banks[i][0], N - 1)
            else:
                w[i, 1 :] = self.bandpass(banks[i][0], banks[i][1], N - 1)
        self.register_buffer("w", w[:, None, :])  # (m, 1, kernel_size)

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

        N = self.bandpass_filter_len - 1
        v = torch.empty((n_banks, N))  # (k, m)
        for i in range(n_banks):
            if i == 0:
                v[i] = self.lowpass(banks[i][1], N)
            elif i == n_banks - 1:
                v[i] = self.highpass(banks[i][0], N)
            else:
                v[i] = self.bandpass(banks[i][0], banks[i][1], N)
        self.register_buffer("v", v[:, None, :])  # (m, 1, kernel_size)

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
        x = real_to_analytic(x)  # (b, c, l)
        x0 = rearrange(x, 'b c l -> (b c) 1 l')

        a2 = fft_convolve_polyphase(x0, self.w, self.factor)

        if True:
            a3 = fftconvolve_complex(x0, self.v, mode="same")
            a3 = a3[:, :, ::self.factor]

        (a2 - a3).abs().mean()
        # from IPython import embed; embed(using=False); os._exit(0)
        
        x = bandpass_demodulate_downsample(
            x=rearrange(x0, 'b c l -> (b c) 1 l'),  # (b*c, 1, l)
            h=self.w[:, :, 1:], 
            sr=sr, 
            freq=-self.f_center, 
            factor=self.factor, 
            chunk_size=self.chunk_size
        )  # (b*c, k, l_out)
        out1 = rearrange(x, '(b c) k l -> b c k l', b=B)

        x = bandpass_demodulate_downsample2(
            x=rearrange(x0, 'b c l -> (b c) 1 l'),  # (b*c, 1, l)
            h=self.w, 
            sr=sr, 
            freq=-self.f_center, 
            factor=self.factor, 
            chunk_size=self.chunk_size
        )  # (b*c, k, l_out)
        out2 = rearrange(x, '(b c) k l -> b c k l', b=B)

        # (out1 - out2).abs().mean()

        # from IPython import embed; embed(using=False); os._exit(0)

        return out2
    

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
        x0 = x

        x1 = upsample_modulate_sum(
            x=rearrange(x0, 'b c k l -> (b c) k l'), 
            up=self.up[1:], 
            sr=sr, 
            freq=self.f_center, 
            factor=self.factor, 
            chunk_size=self.chunk_size
        )

        x2 = upsample_modulate_sum2(
            x=rearrange(x0, 'b c k l -> (b c) k l'), 
            up=self.up, 
            sr=sr, 
            freq=self.f_center, 
            factor=self.factor, 
            chunk_size=self.chunk_size
        )
        # print((x1 - x2).abs().mean())
        # from IPython import embed; embed(using=False); os._exit(0)

        x = rearrange(x2, '(b c) l -> b c l', b=B)
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



def fft_convolve_polyphase(x: Tensor, w: Tensor, stride: int) -> Tensor:
    r"""Convolution with polylphase.

    b: batch_size
    k: n_banks
    l: audio_samples
    n: filter_len

    Args:
        x: (b, k, l)
        w: (b, 1, n)

    Returns:
        out: (b, k, l_down)
    """
    assert torch.all(w[:, :, 0] == 0)
    x = rearrange(x, 'b 1 (t1 t2) -> b t2 t1', t2=stride)  # (b, t2, t1)
    w = rearrange(w, 'k 1 (n1 n2) -> k n2 n1', n2=stride)  # (b, n2, n1)
    
    L = x.shape[-1]
    N = w.shape[-1]
    assert N % 2 == 0

    w = torch.flip(w, dims=[2])  # (b, n2, n1)
    x = fftconvolve_complex(x, w, mode="full")  # (b, k, l_down)
    out = fix_length(x, N // 2 - 1, L)  # (b, k, l_down)

    return out


def fft_upsample_polyphase(x: Tensor, w: Tensor, stride: int) -> Tensor:
    r"""Convolution with polylphase.

    b: batch_size
    k: n_banks
    l: audio_samples
    n: filter_len

    Args:
        x: (b, l_down)
        w: (n,)

    Returns:
        out: (b, 1, l_up)
    """
    # from IPython import embed; embed(using=False); os._exit(0)
    L = x.shape[-1] * stride
    N = w.shape[-1]
    x = rearrange(x, 'b l -> b 1 l')
    w = rearrange(w, '(n1 n2) -> n2 1 n1', n2=stride)  # (b, n2, n1)
    # x = fftconvolve_complex(x, w, mode="same")  # (b, k, l_down)
    x = fftconvolve_complex(x, w, mode="full")  # (b, k, l_down)
    x = rearrange(x, 'b t2 t1 -> b (t1 t2)')
    out = fix_length(x, N // 2, L)
    # from IPython import embed; embed(using=False); os._exit(0)
    
    return out


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
        # from IPython import embed; embed(using=False); os._exit(0)
        
        # Demodulate
        e = shift_frequency(e, freq[k : k + chunk_size], sr)  # (b, s, l_in)

        # Downsample
        out[:, k : k + chunk_size, :] = e[:, :, 0 :: factor]  # (b, 1, l_out)

        k += chunk_size

    # from IPython import embed; embed(using=False); os._exit(0)

    return out


def bandpass_demodulate_downsample2(
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
        e = fft_convolve_polyphase(x, h[k : k + chunk_size, :, :], factor)  # (b, s, l_in)

        # e2 = fftconvolve_complex(x, h[k : k + chunk_size, :, 1:], mode="same")[:, :, ::factor]
        # from IPython import embed; embed(using=False); os._exit(0)
        
        # Demodulate
        e = shift_frequency(e, freq[k : k + chunk_size], sr, factor)  # (b, s, l_in)
        tmp = e

        # Downsample
        out[:, k : k + chunk_size, :] = e  # (b, 1, l_out)

        k += chunk_size

        # from IPython import embed; embed(using=False); os._exit(0)
        # e2 = fftconvolve_complex(x, h[k : k + chunk_size, :, 1:], mode="same")
        # e2 = shift_frequency(e2, freq[k : k + chunk_size], sr)  # (b, s, l_in)
        # e2 = e2[:, :, ::factor]
        # from IPython import embed; embed(using=False); os._exit(0)

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

        # from IPython import embed; embed(using=False); os._exit(0)
        k += chunk_size
        

    return out


def upsample_modulate_sum2(
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
        e1 = torch.zeros(B, min(chunk_size, K - k), L, dtype=x.dtype, device=x.device)
        e1[:, :, ::factor] = x[:, k : k + chunk_size, :]  # (b, s, l)
        e1 = rearrange(e1, 'b s l -> (b s) 1 l')  # (b*s, 1, l)
        z1 = fftconvolve_complex(e1 * factor, up[None, None, 1:])
        z1 = rearrange(z1, '(b s) 1 l -> b s l', b=B) 

        e2 = x[:, k : k + chunk_size, :]
        e2 = rearrange(e2, 'b s l -> (b s) l')  # (b*s, 1, l)
        z2 = fft_upsample_polyphase(e2 * factor, up, factor)
        z2 = rearrange(z2, '(b s) l -> b s l', b=B) 
        # print((z1 - z2).abs().mean())
        # from IPython import embed; embed(using=False); os._exit(0)

        # e = fftconvolve_complex(e * factor, up[None, None, :])  # (b*s, 1, l)
        # e = rearrange(z2, '(b s) l -> b s l', b=B)  # (b, s, l) 

        # # Modulate
        e = shift_frequency(z2, freq[k : k + chunk_size], sr)  # (b, s, l)

        # # Sum
        out.add_(e.sum(dim=1))  # (b, l)
        # from IPython import embed; embed(using=False); os._exit(0)

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
    out = x * a
    return out
    # x.mul_(a)
    # return x


def to_same(x, N, length):
    start = (N - 1) // 2
    end = start + length
    return x[..., start : end]

def fix_length(x, start, length):
    end = start + length
    return x[..., start : end]


if __name__ == '__main__':
    
    sr = 48000
    # n_bands = 256
    n_bands = 64
    max_bandwidth = 800
    chunk_size = 16  # Try to tune this to balance RAM and computation speed
    # factor = sr // max_bandwidth
    factor = 10
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