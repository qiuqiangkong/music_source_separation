import math
import time

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from scipy.signal import firwin
from torch import Tensor

from mss.models2.dsp.analytic import analytic_to_real, real_to_analytic
from mss.models2.dsp.banks import mel_linear_banks
from mss.models2.dsp.convolve import fftconvolve, fftconvolve_complex
# from mss.models2.dsp.resample import UpSample
from mss.models2.dsp.upsample2 import UpSample, UpSample3
from mss.utils import fast_sdr


'''
class SubbandFilter(nn.Module):
    r"""Save memory version. Split signal into subbands."""

    def __init__(
        self, 
        sr: int, 
        banks: list[tuple[int, int]],
        filter_len: int = 10001,
    ):
        r"""
        k: n_bands
        m: filter_len
        """

        super().__init__()

        # filter_len = 10000

        self.sr = sr
        self.banks = banks
        self.filter_len = filter_len
        self.window_type = "hamming"
        self.factor = 10

        n_banks = len(banks)
        w = torch.empty((n_banks, self.filter_len))  # (k, m)

        for i in range(n_banks):
            if i == 0:
                w[i] = self.lowpass(banks[i][1])
            elif i == n_banks - 1:
                w[i] = self.highpass(banks[i][0])
            else:
                w[i] = self.bandpass(banks[i][0], banks[i][1])

        self.register_buffer("w", w[:, None, :])  # (m, 1, kernel_size)

        bandwidths = [bank[1] - bank[0] for bank in banks]  # (k,)
        assert max(bandwidths) <= sr / self.factor

        self.register_buffer("f_center", Tensor([np.mean(bank) for bank in banks]))  # (k,)

        self.upsample = UpSample(self.factor, 99)
        # from IPython import embed; embed(using=False); os._exit(0)

        # self.upsample2 = UpSample2()


        self.w2 = torch.empty((n_banks, self.filter_len, self.factor))  # (k, m, r)

    def lowpass(self, f: float) -> Tensor:
        h = firwin(
            numtaps=self.filter_len, 
            cutoff=f / (self.sr / 2), 
            pass_zero="lowpass",
            window=self.window_type
        )
        return torch.from_numpy(h)  # (m,)

    def bandpass(self, f1: float, f2: float) -> Tensor:
        h = firwin(
            numtaps=self.filter_len, 
            cutoff=[f1 / (self.sr / 2), f2 / (self.sr / 2)], 
            pass_zero="bandpass",
            window=self.window_type
        )
        return torch.from_numpy(h)  # (m,)

    def highpass(self, f: float) -> Tensor:
        h = firwin(
            numtaps=self.filter_len, 
            cutoff=f / (self.sr / 2), 
            pass_zero="highpass",
            window=self.window_type
        )
        return torch.from_numpy(h)  # (m,)

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

        x0 = x
        x = real_to_analytic(x)  # (b, c, l)

        B = x.shape[0]
        x = rearrange(x, 'b c l -> (b c) 1 l')  # (b*c, 1, l)
        x = fftconvolve(x.real, self.w) + 1.j * fftconvolve(x.imag, self.w)  # (b*c, k, l)

        x = x[:, :, ::self.factor]
        x = self.shift_frequency(x, -self.f_center, self.factor)  # (b*c, k, l_in)

        # from IPython import embed; embed(using=False); os._exit(0)
        
        
        # import matplotlib.pyplot as plt
        # y = torch.fft.fft(x)
        # plt.plot(torch.abs(y)[0, 65, :].cpu().numpy())
        # plt.savefig("_zz.pdf")
        # from IPython import embed; embed(using=False); os._exit(0)

        latent = x

        # # Decode
        # x = self.shift_frequency(x, self.f_center, 1.)  # (b, c, k, l_in)
        K = x.shape[1]
        x = rearrange(x, '(b c) k l -> (b c k) l', b=B)
        x = self.upsample(x.real) + 1.j * self.upsample(x.imag)
        x = rearrange(x, '(b c k) l -> (b c) k l', b=B, k=K)
        x = self.shift_frequency(x, self.f_center, 1.)  # (b, c, k, l_in)

        
        x = rearrange(x, '(b c) k l -> b c k l', b=B)
        x = x.sum(2)  # (b, l)
        x = analytic_to_real(x)

        # print((x-x0).abs().mean())
        print(fast_sdr(x0.cpu().numpy(), x.cpu().numpy()))

        from IPython import embed; embed(using=False); os._exit(0)

        # x: (b, k, l'), as_stride: (b, k, n_frames, m)
        # w: (m,)
        # out: (b, l), as_stride: (b, n_frames, m)
        
        
        return rearrange(x, '(b c) k l -> b c k l', b=B)  # (b, c, k, l)

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
        return x.sum(2)

    def shift_frequency(self, x: Tensor, f: Tensor) -> Tensor:
        r"""Shift frequency to -ω0.

        X(j(ω-ω0)) = e^{j(ω0)n}X(jω)

        Args:
            x: (any, k, l)
            f: (k,)

        Returns:
            out: (any, k, l)
        """
        omega = f / (self.sr / 2) * math.pi  # (k,)
        t = torch.arange(x.shape[-1], device=x.device)  # (l,)
        a = torch.exp(1.j * omega[:, None] * t[None, :])
        x.mul_(a)
        return x

    def shift_frequency(self, x: Tensor, f: Tensor, n=1) -> Tensor:
        r"""Shift frequency to -ω0.

        X(j(ω-ω0)) = e^{j(ω0)n}X(jω)

        Args:
            x: (any, k, l)
            f: (k,)

        Returns:
            out: (any, k, l)
        """
        omega = f / (self.sr / 2) * math.pi  # (k,)
        t = torch.arange(x.shape[-1], device=x.device) * n  # (l,)
        a = torch.exp(1.j * omega[:, None] * t[None, :])
        x.mul_(a)
        return x
'''

'''
class SubbandFilter(nn.Module):
    r"""Save memory version. Split signal into subbands."""

    def __init__(
        self, 
        sr: int, 
        banks: list[tuple[int, int]],
        filter_len: int = 10001,
    ):
        r"""
        k: n_bands
        m: filter_len
        """

        super().__init__()

        filter_len = 10000

        self.sr = sr
        self.banks = banks
        self.filter_len = filter_len
        self.window_type = "hamming"
        self.factor = 10

        n_banks = len(banks)
        w = torch.empty((n_banks, self.filter_len))  # (k, m)

        for i in range(n_banks):
            if i == 0:
                w[i] = self.lowpass(banks[i][1])
            elif i == n_banks - 1:
                w[i] = self.highpass(banks[i][0])
            else:
                w[i] = self.bandpass(banks[i][0], banks[i][1])

        self.register_buffer("w", w[:, None, :])  # (m, 1, kernel_size)

        bandwidths = [bank[1] - bank[0] for bank in banks]  # (k,)
        assert max(bandwidths) <= sr / self.factor

        self.register_buffer("f_center", Tensor([np.mean(bank) for bank in banks]))  # (k,)

        self.upsample = UpSample(self.factor, 200)
        # from IPython import embed; embed(using=False); os._exit(0)

        # self.upsample2 = UpSample2()


        self.w2 = torch.empty((n_banks, self.filter_len, self.factor))  # (k, m, r)

    def lowpass(self, f: float) -> Tensor:
        h = firwin(
            numtaps=to_odd(self.filter_len), 
            cutoff=f / (self.sr / 2), 
            pass_zero="lowpass",
            window=self.window_type
        )
        return torch.from_numpy(pad_filter_to_even(h))  # (n,)

    def bandpass(self, f1: float, f2: float) -> Tensor:
        h = firwin(
            numtaps=to_odd(self.filter_len), 
            cutoff=[f1 / (self.sr / 2), f2 / (self.sr / 2)], 
            pass_zero="bandpass",
            window=self.window_type
        )
        return torch.from_numpy(pad_filter_to_even(h))  # (n,)

    def highpass(self, f: float) -> Tensor:
        h = firwin(
            numtaps=to_odd(self.filter_len), 
            cutoff=f / (self.sr / 2), 
            pass_zero="highpass",
            window=self.window_type
        )
        return torch.from_numpy(pad_filter_to_even(h))  # (n,)

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

        # from IPython import embed; embed(using=False); os._exit(0)
        x0 = x
        x = real_to_analytic(x)  # (b, c, l)

        B = x.shape[0]
        x = rearrange(x, 'b c l -> (b c) 1 l')  # (b*c, 1, l)
        x = fftconvolve(x.real, self.w) + 1.j * fftconvolve(x.imag, self.w)  # (b*c, k, l)

        x = x[:, :, ::self.factor]
        x = self.shift_frequency(x, -self.f_center, self.factor)  # (b*c, k, l_in)

        
        
        
        # import matplotlib.pyplot as plt
        # y = torch.fft.fft(x)
        # plt.plot(torch.abs(y)[0, 65, :].cpu().numpy())
        # plt.savefig("_zz.pdf")
        # from IPython import embed; embed(using=False); os._exit(0)

        latent = x

        # # Decode
        # x = self.shift_frequency(x, self.f_center, 1.)  # (b, c, k, l_in)
        K = x.shape[1]
        x = rearrange(x, '(b c) k l -> (b c k) l', b=B)
        # x = self.upsample(x.real) + 1.j * self.upsample(x.imag)
        x = self.upsample(x)
        x = rearrange(x, '(b c k) l -> (b c) k l', b=B, k=K)
        x = self.shift_frequency(x, self.f_center, 1.)  # (b, c, k, l_in)

        
        x = rearrange(x, '(b c) k l -> b c k l', b=B)
        x = x.sum(2)  # (b, l)
        x = analytic_to_real(x)

        # print((x-x0).abs().mean())
        print(fast_sdr(x0.cpu().numpy(), x.cpu().numpy()))

        from IPython import embed; embed(using=False); os._exit(0)

        # x: (b, k, l'), as_stride: (b, k, n_frames, m)
        # w: (m,)
        # out: (b, l), as_stride: (b, n_frames, m)
        
        
        return rearrange(x, '(b c) k l -> b c k l', b=B)  # (b, c, k, l)

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
        return x.sum(2)

    def shift_frequency(self, x: Tensor, f: Tensor) -> Tensor:
        r"""Shift frequency to -ω0.

        X(j(ω-ω0)) = e^{j(ω0)n}X(jω)

        Args:
            x: (any, k, l)
            f: (k,)

        Returns:
            out: (any, k, l)
        """
        omega = f / (self.sr / 2) * math.pi  # (k,)
        t = torch.arange(x.shape[-1], device=x.device)  # (l,)
        a = torch.exp(1.j * omega[:, None] * t[None, :])
        x.mul_(a)
        return x

    def shift_frequency(self, x: Tensor, f: Tensor, n=1) -> Tensor:
        r"""Shift frequency to -ω0.

        X(j(ω-ω0)) = e^{j(ω0)n}X(jω)

        Args:
            x: (any, k, l)
            f: (k,)

        Returns:
            out: (any, k, l)
        """
        omega = f / (self.sr / 2) * math.pi  # (k,)
        t = torch.arange(x.shape[-1], device=x.device) * n  # (l,)
        a = torch.exp(1.j * omega[:, None] * t[None, :])
        x.mul_(a)
        return x
'''

class SubbandFilter(nn.Module):
    r"""Save memory version. Split signal into subbands."""

    def __init__(
        self, 
        sr: int, 
        banks: list[tuple[int, int]],
        filter_len: int = 10001,
    ):
        r"""
        k: n_bands
        m: filter_len
        """

        super().__init__()

        # filter_len = 10000

        self.sr = sr
        self.banks = banks
        self.filter_len = filter_len
        self.window_type = "hamming"
        self.factor = 10

        self.filter_len = 10001

        n_banks = len(banks)
        w = torch.empty((n_banks, self.filter_len))  # (k, m)

        for i in range(n_banks):
            if i == 0:
                w[i] = self.lowpass(banks[i][1])
            elif i == n_banks - 1:
                w[i] = self.highpass(banks[i][0])
            else:
                w[i] = self.bandpass(banks[i][0], banks[i][1])

        self.register_buffer("w", w[:, None, :])  # (m, 1, kernel_size)

        bandwidths = [bank[1] - bank[0] for bank in banks]  # (k,)
        assert max(bandwidths) <= sr / self.factor

        self.register_buffer("f_center", Tensor([np.mean(bank) for bank in banks]))  # (k,)

        self.upsample = UpSample(self.factor, 200)

        up = Tensor(firwin(
            numtaps=201, 
            cutoff=1. / factor, 
            pass_zero="lowpass",
            window="hamming"
        )) * self.factor
        self.register_buffer("up", up)


    def lowpass(self, f: float) -> Tensor:
        h = firwin(
            numtaps=self.filter_len, 
            cutoff=f / (self.sr / 2), 
            pass_zero="lowpass",
            window=self.window_type
        )
        return torch.from_numpy(h)  # (n,)

    def bandpass(self, f1: float, f2: float) -> Tensor:
        h = firwin(
            numtaps=self.filter_len, 
            cutoff=[f1 / (self.sr / 2), f2 / (self.sr / 2)], 
            pass_zero="bandpass",
            window=self.window_type
        )
        return torch.from_numpy(h)  # (n,)

    def highpass(self, f: float) -> Tensor:
        h = firwin(
            numtaps=to_odd(self.filter_len), 
            cutoff=f / (self.sr / 2), 
            pass_zero="highpass",
            window=self.window_type
        )
        return torch.from_numpy(h)  # (n,)

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

        x0 = x
        x = real_to_analytic(x)  # (b, c, l)

        B = x.shape[0]
        x = rearrange(x, 'b c l -> (b c) 1 l')  # (b*c, 1, l)

        x = bandpass_demodulate_downsample(
            x=x, 
            h=self.w, 
            sr=sr, 
            freq=-self.f_center, 
            factor=self.factor, 
            chunk_size=4
        )  # (b*c, k, l_out)
        

        y = upsample_modulate_sum(
            x=x, 
            up=self.up, 
            sr=sr, 
            freq=self.f_center, 
            factor=self.factor, 
            chunk_size=4
        )

        y = rearrange(y, '(b c) l -> b c l', b=B)

        

        # x = rearrange(x, '(b c) k l -> (b c k) l', b=B)
        # x = self.upsample(x)
        # x = rearrange(x, '(b c k) l -> (b c) k l', b=B, c=2)

        # from IPython import embed; embed(using=False); os._exit(0)
        
        # latent = x
        
        # # Decode
        K = x.shape[1]
        x = rearrange(x, '(b c) k l -> (b c k) l', b=B)
        x = self.upsample(x)
        x = rearrange(x, '(b c k) l -> (b c) k l', b=B, k=K)

        x = self.shift_frequency(x, self.f_center, 1.)  # (b, c, k, l_in)

        
        x = rearrange(x, '(b c) k l -> b c k l', b=B)
        x = x.sum(2)  # (b, l)
        
        x = analytic_to_real(x)

        print(fast_sdr(x0.cpu().numpy(), y.cpu().numpy()))
        from IPython import embed; embed(using=False); os._exit(0)
        
        # return rearrange(x, '(b c) k l -> b c k l', b=B)  # (b, c, k, l)
        

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
        return x.sum(2)

    def shift_frequency(self, x: Tensor, f: Tensor) -> Tensor:
        r"""Shift frequency to -ω0.

        X(j(ω-ω0)) = e^{j(ω0)n}X(jω)

        Args:
            x: (any, k, l)
            f: (k,)

        Returns:
            out: (any, k, l)
        """
        omega = f / (self.sr / 2) * math.pi  # (k,)
        t = torch.arange(x.shape[-1], device=x.device)  # (l,)
        a = torch.exp(1.j * omega[:, None] * t[None, :])
        x.mul_(a)
        return x

    def shift_frequency(self, x: Tensor, f: Tensor, n=1) -> Tensor:
        r"""Shift frequency to -ω0.

        X(j(ω-ω0)) = e^{j(ω0)n}X(jω)

        Args:
            x: (any, k, l)
            f: (k,)

        Returns:
            out: (any, k, l)
        """
        omega = f / (self.sr / 2) * math.pi  # (k,)
        t = torch.arange(x.shape[-1], device=x.device) * n  # (l,)
        a = torch.exp(1.j * omega[:, None] * t[None, :])
        x.mul_(a)
        return x


# @torch.compile
def bandpass_demodulate_downsample(
    x: Tensor, 
    h: Tensor, 
    sr: int, 
    freq: Tensor, 
    factor: int,
    chunk_size: int
) -> Tensor:
    r"""

    Args:
        x: (b, 1, l_in)
        h: (k, 1, n)
        sr: int
        freq: (k,)
        factor: int

    Returns:
        out: (b, k, l_out)
    """
    B = x.shape[0]
    K = h.shape[0]
    L = math.ceil(x.shape[2] / factor)

    out = torch.empty(B, K, L, dtype=x.dtype, device=x.device)  # (b, k, l_out)
    k = 0

    while k < K:
        e = fftconvolve_complex(x, h[k : k + chunk_size, :, :])  # (b, s, l_in)
        e = shift_frequency(e, freq[k : k + chunk_size], sr)  # (b, s, l_in)
        out[:, k : k + chunk_size, :] = e[:, :, 0 :: factor]  # (b, s, l_out)
        k += chunk_size

    return out


'''
def upsample_modulate_sum(
    x: Tensor, 
    up: Tensor, 
    sr: int, 
    freq: Tensor, 
    factor: int,
    chunk_size: int
):
    r"""

    Args:
        x: (b, k, l')
        up: (n,)
    """

    B, K = x.shape[0 : 2]
    L = x.shape[2] * factor
    
    out = torch.zeros(B, L, dtype=x.dtype, device=x.device)  # (b, l)
    k = 0

    while k < K:
        
        # Upsample
        e = torch.zeros(B, chunk_size, L, dtype=x.dtype, device=x.device)
        e[:, :, ::factor] = x[:, k : k + chunk_size, :]
        # from IPython import embed; embed(using=False); os._exit(0)
        e = rearrange(e, 'b s l -> (b s) 1 l')  # (b*s, 1, l)
        e = fftconvolve_complex(e, up[None, None, :])  # (b*s, 1, l)
        e = rearrange(e, '(b s) 1 l -> b s l', b=B)  # (b, s, 1, l)

        # Modulate
        e = shift_frequency(e, freq[k : k + chunk_size], sr)
        
        # Sum
        out.add_(e.sum(dim=1))  # (b, l)

        k += chunk_size

    return out
'''

def upsample_modulate_sum(
    x: Tensor, 
    up: Tensor, 
    sr: int, 
    freq: Tensor, 
    factor: int,
    chunk_size: int
):
    r"""

    Args:
        x: (b, k, l')
        up: (n,)
    """

    B, K = x.shape[0 : 2]
    L = x.shape[2] * factor
    
    out = torch.zeros(B, L, dtype=x.dtype, device=x.device)  # (b, l)
    k = 0

    while k < K:
        
        # Upsample
        e = torch.zeros(B, chunk_size, L, dtype=x.dtype, device=x.device)
        e[:, :, ::factor] = x[:, k : k + chunk_size, :]
        # from IPython import embed; embed(using=False); os._exit(0)
        e = rearrange(e, 'b s l -> (b s) 1 l')  # (b*s, 1, l)
        e = fftconvolve_complex(e, up[None, None, :])  # (b*s, 1, l)
        e = rearrange(e, '(b s) 1 l -> b s l', b=B)  # (b, s, 1, l)

        # # Modulate
        # e = shift_frequency(e, freq[k : k + chunk_size], sr)
        
        # # Sum
        out.add_(e.sum(dim=1))  # (b, l)

        k += chunk_size

    return out



def shift_frequency(x: Tensor, f: Tensor, sr: int, n=1) -> Tensor:
    r"""Shift frequency to -ω0.

    X(j(ω-ω0)) = e^{j(ω0)n}X(jω)

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


'''
class SubbandFilter(nn.Module):
    r"""Save memory version. Split signal into subbands."""

    def __init__(
        self, 
        sr: int, 
        banks: list[tuple[int, int]],
        filter_len: int = 10001,
    ):
        r"""
        k: n_bands
        m: filter_len
        """

        super().__init__()

        filter_len = 10000

        self.sr = sr
        self.banks = banks
        self.filter_len = filter_len
        self.window_type = "hamming"
        self.factor = 10

        n_banks = len(banks)
        w = torch.empty((n_banks, self.filter_len))  # (k, m)

        for i in range(n_banks):
            if i == 0:
                w[i] = self.lowpass(banks[i][1])
            elif i == n_banks - 1:
                w[i] = self.highpass(banks[i][0])
            else:
                w[i] = self.bandpass(banks[i][0], banks[i][1])

        self.register_buffer("w", w[:, None, :])  # (m, 1, kernel_size)

        bandwidths = [bank[1] - bank[0] for bank in banks]  # (k,)
        assert max(bandwidths) <= sr / self.factor

        self.register_buffer("f_center", Tensor([np.mean(bank) for bank in banks]))  # (k,)

        # from IPython import embed; embed(using=False); os._exit(0)
        omega = self.f_center / (self.sr / 2) * math.pi  # (k,)
        t = torch.arange(self.factor)  # (l,)
        a = torch.exp(1.j * omega[:, None] * t[None, :])  # (k, r)
        # from IPython import embed; embed(using=False); os._exit(0)

        self.upsample = UpSample3(self.factor, 200, n_banks, a)
        # from IPython import embed; embed(using=False); os._exit(0)

        # self.upsample2 = UpSample2()


        self.w2 = torch.empty((n_banks, self.filter_len, self.factor))  # (k, m, r)

    def lowpass(self, f: float) -> Tensor:
        h = firwin(
            numtaps=to_odd(self.filter_len), 
            cutoff=f / (self.sr / 2), 
            pass_zero="lowpass",
            window=self.window_type
        )
        return torch.from_numpy(pad_filter_to_even(h))  # (n,)

    def bandpass(self, f1: float, f2: float) -> Tensor:
        h = firwin(
            numtaps=to_odd(self.filter_len), 
            cutoff=[f1 / (self.sr / 2), f2 / (self.sr / 2)], 
            pass_zero="bandpass",
            window=self.window_type
        )
        return torch.from_numpy(pad_filter_to_even(h))  # (n,)

    def highpass(self, f: float) -> Tensor:
        h = firwin(
            numtaps=to_odd(self.filter_len), 
            cutoff=f / (self.sr / 2), 
            pass_zero="highpass",
            window=self.window_type
        )
        return torch.from_numpy(pad_filter_to_even(h))  # (n,)

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

        # from IPython import embed; embed(using=False); os._exit(0)
        x0 = x
        x = real_to_analytic(x)  # (b, c, l)

        B = x.shape[0]
        x = rearrange(x, 'b c l -> (b c) 1 l')  # (b*c, 1, l)
        x = fftconvolve(x.real, self.w) + 1.j * fftconvolve(x.imag, self.w)  # (b*c, k, l)

        x = x[:, :, ::self.factor]
        x = self.shift_frequency(x, -self.f_center, self.factor)  # (b*c, k, l_in)

        
        
        
        # import matplotlib.pyplot as plt
        # y = torch.fft.fft(x)
        # plt.plot(torch.abs(y)[0, 65, :].cpu().numpy())
        # plt.savefig("_zz.pdf")
        # from IPython import embed; embed(using=False); os._exit(0)

        latent = x

        # # Decode
        
        K = x.shape[1]
        # x = rearrange(x, '(b c) k l -> (b c k) l', b=B)


        # from IPython import embed; embed(using=False); os._exit(0)

        x = self.shift_frequency(x, self.f_center, self.factor)  # (b*c, k, l_in)

        x = self.upsample(x)
        x = rearrange(x, '(b c) l -> b c l', b=B)
        # x = self.shift_frequency(x, self.f_center, 1.)  # (b, c, k, l_in)

        
        # x = rearrange(x, '(b c) k l -> b c k l', b=B)
        # x = x.sum(2)  # (b, l)
        x = analytic_to_real(x)
        # x = rearrange(x, '(b c k) ')

        # print((x-x0).abs().mean())
        print(fast_sdr(x0.cpu().numpy(), x.cpu().numpy()))

        from IPython import embed; embed(using=False); os._exit(0)

        # x: (b, k, l'), as_stride: (b, k, n_frames, m)
        # w: (m,)
        # out: (b, l), as_stride: (b, n_frames, m)
        
        
        return rearrange(x, '(b c) k l -> b c k l', b=B)  # (b, c, k, l)

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
        return x.sum(2)

    def shift_frequency(self, x: Tensor, f: Tensor) -> Tensor:
        r"""Shift frequency to -ω0.

        X(j(ω-ω0)) = e^{j(ω0)n}X(jω)

        Args:
            x: (any, k, l)
            f: (k,)

        Returns:
            out: (any, k, l)
        """
        omega = f / (self.sr / 2) * math.pi  # (k,)
        t = torch.arange(x.shape[-1], device=x.device)  # (l,)
        a = torch.exp(1.j * omega[:, None] * t[None, :])
        x.mul_(a)
        return x

    def shift_frequency(self, x: Tensor, f: Tensor, n=1) -> Tensor:
        r"""Shift frequency to -ω0.

        X(j(ω-ω0)) = e^{j(ω0)n}X(jω)

        Args:
            x: (any, k, l)
            f: (k,)

        Returns:
            out: (any, k, l)
        """
        omega = f / (self.sr / 2) * math.pi  # (k,)
        t = torch.arange(x.shape[-1], device=x.device) * n  # (l,)
        a = torch.exp(1.j * omega[:, None] * t[None, :])
        x.mul_(a)
        return x
'''

def to_odd(N):
    if N % 2 == 0:
        return N - 1
    else:
        return N


def pad_filter_to_even(h):
    if h.shape[-1] % 2 == 1:
        return np.pad(h, (0, 1))  # (n,)
    else:
        return h
    

class SubbandResampler(nn.Module):
    def __init__(
        self, 
        sr: int,
        banks: list[tuple[float, float]],
        factor: int,
        filter_len=10001
    ):
        super().__init__()

        self.sr = sr
        self.factor = factor
        self.register_buffer("f_center", Tensor([np.mean(bank) for bank in banks]))  # (k,)

        bandwidths = [bank[1] - bank[0] for bank in banks]  # (k,)
        assert max(bandwidths) <= sr / self.factor

        self.upsample = UpSample(self.factor, filter_len)

    def analysis(self, x: Tensor) -> Tensor:
        r"""Downsample subband signals.

        b: batch_size
        c: audio_channels
        k: n_bands

        Args:
            x: (b, c, k, l_in)

        Returns:
            out: (b, c, k, l_out)
        """
        
        # Get analytic signal
        x = real_to_analytic(x)  # (b, c, k, l_in)

        # Move ω0 to center
        x = self.shift_frequency(x, -self.f_center)  # (b, c, k, l_in)

        # Downsample
        x = x[:, :, :, 0 :: self.factor]  # (b, c, k, l_out)

        return x

    def synthesis(self, x: Tensor) -> Tensor:
        r"""Upsample subband signals.

        b: batch_size
        c: audio_channels
        k: n_bands

        Args:
            x: (b, c, k, l_in)

        Returns:
            out: (b, c, k, l_out)
        """
        
        B, C, K, L = x.shape

        # Upsample
        x = rearrange(x, 'b c k l -> (b c k) l')
        x = self.upsample(x.real) + 1.j * self.upsample(x.imag)
        x = rearrange(x, '(b c k) l -> b c k l', b=B, c=C)
        
        # Move back to ω0
        x = self.shift_frequency(x, self.f_center)

        # Analytic to original signal
        x = analytic_to_real(x)

        return x

    def shift_frequency(self, x: Tensor, f: Tensor) -> Tensor:
        r"""Shift frequency to -ω0.

        X(j(ω-ω0)) = e^{j(ω0)n}X(jω)

        Args:
            x: (any, k, l)
            f: (k,)

        Returns:
            out: (any, k, l)
        """
        omega = f / (self.sr / 2) * math.pi  # (k,)
        t = torch.arange(x.shape[-1], device=x.device)  # (l,)
        a = torch.exp(1.j * omega[:, None] * t[None, :])
        x.mul_(a)
        return x


if __name__ == '__main__':
    
    sr = 48000
    n_bands = 64
    max_bandwidth = 800
    filter_len = 10001
    factor = sr // max_bandwidth
    device = "cuda"
    
    # Melbanks
    banks = mel_linear_banks(sr=sr, n_bands=n_bands, max_bandwidth=max_bandwidth)
    sb_filter = SubbandFilter(sr, banks, filter_len).to(device)
    # sb_resampler = SubbandResampler(sr, banks, factor, filter_len).to(device)

    # for _ in range(20):
    while True:

        # Audio
        rs = np.random.RandomState(1234)
        audio = rs.uniform(low=-1, high=1, size=(4, 2, sr * 2))
        audio = Tensor(audio).to(device)  # (c, l)
        
        # Analysis
        t0 = time.time()
        x = sb_filter.analysis(audio)  # (b, c, k, l)
        # latent = sb_resampler.analysis(x)
        
        # Synthesis
        # y = sb_resampler.synthesis(latent)
        # y = sb_filter.synthesis(y)
        
        # Print
        t1 = time.time() - t0
        print(t1)
        # sdr = fast_sdr(audio.cpu().numpy(), y.cpu().numpy())
        # print(f"time: {t1:.4f} s, latent: {latent.shape}, SDR: {sdr:.2f} dB")