from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from mss.models.attention import Block
from mss.models2.bandsplit42a import BandSplit
from mss.models2.bases_wave import transform as bases_wave_transform
from mss.models2.bases_wave import inverse_transform as bases_wave_inverse_transform
from mss.models.fourier import Fourier
from mss.models.rope import RoPE
import time



class BSRoformer80a(nn.Module):
    def __init__(
        self,
        audio_channels=2,
        sample_rate=48000,
        n_fft=2048,
        hop_length=480,
        n_bands=256,
        band_dim=64,
        patch_size=[4, 4],
        n_layers=12,
        n_heads=12,
        dim=768,
        rope_len=8192,
        **kwargs
    ) -> None:

        super().__init__()

        self.ac = audio_channels
        self.patch_size = patch_size

        # Band split
        self.bandsplit = BandSplit(
            sr=sample_rate, 
            n_fft=n_fft, 
            n_bands=n_bands,
            in_channels=2,  # real + imag
            out_channels=band_dim
        )

        self.fourier_2048 = Fourier(n_fft=2048, hop_length=480, return_complex=True, normalized=True)
        self.patch_2048 = Patch(band_dim * audio_channels, dim, (4, 4))
        self.unpatch_2048 = UnPatch(dim, band_dim * audio_channels, (4, 4))

        #
        self.bases_wave = BasesWave(n_window=2048, hop_length=480)
        self.patch_wave = Patch2(2, dim, (4, 32))
        self.unpatch_wave = UnPatch2(dim, 2, (4, 32))

        #
        self.pre_fc = nn.Linear(dim*2, dim)
        self.post_fc = nn.Linear(dim, dim*2)

        # RoPE
        self.rope = RoPE(head_dim=dim // n_heads, max_len=rope_len)

        # Transformer blocks
        self.t_blocks = nn.ModuleList(Block(dim, n_heads) for _ in range(n_layers))
        self.f_blocks = nn.ModuleList(Block(dim, n_heads) for _ in range(n_layers))

    def forward(self, audio: Tensor) -> Tensor:
        r"""Separation model.

        b: batch_size
        c: channels_num
        l: audio_samples
        t: frames_num
        f: freq_bins

        Args:
            audio: (b, c, t)

        Outputs:
            output: (b, c, t)
        """

        # --- 1. Encode ---
        # 1.1 Complex spectrum
        sp_2048 = self.fourier_2048.stft(audio)
        T_2048 = sp_2048.shape[2]
        x = torch.view_as_real(sp_2048)  # shape: (b, c, t, f, 2)
        x = self.pad_tensor(x)  # x: (b, d, t, f)
        x = self.bandsplit.transform(x)  # shape: (b, c, t, f, o)
        x = self.patch_2048(x)
        x_2048 = x

        # Wave
        x = self.bases_wave.encode(audio)
        x = self.pad_tensor2(x)  # x: (b, d, t, f)
        x_wave = self.patch_wave(x)
        
        # Concat
        x = torch.cat([x_2048, x_wave], dim=1)
        x = rearrange(x, 'b d t f -> b t f d')
        x = self.pre_fc(x)
        x = rearrange(x, 'b t f d -> b d t f')

        B = x.shape[0]
        # T1 = x.shape[2]

        # --- 2. Transformer along time and frequency axes ---
        for t_block, f_block in zip(self.t_blocks, self.f_blocks):

            x = rearrange(x, 'b d t f -> (b f) t d')
            x = t_block(x, rope=self.rope, pos=None)  # shape: (b*f, t, d)

            x = rearrange(x, '(b f) t d -> (b t) f d', b=B)
            x = f_block(x, rope=self.rope, pos=None)  # shape: (b*t, f, d)

            x = rearrange(x, '(b t) f d -> b d t f', b=B)  # shape: (b, d, t, f)

        x = rearrange(x, 'b d t f -> b t f d')
        x = self.post_fc(x)
        x = rearrange(x, 'b t f d -> b d t f')
        x_2048, x_wave = x.chunk(chunks=2, dim=1)

        #
        x = self.unpatch_2048(x_2048, self.ac)
        x = self.bandsplit.inverse_transform(x)  # shape: (b, c, t, f, k)
        x = x[:, :, 0 : T_2048, :, :]
        mask = torch.view_as_complex(x)  # shape: (b, c, t, f)
        y_2048 = mask * sp_2048  # shape: (b, c, t, f)
        y_2048 = self.fourier_2048.istft(y_2048)  # shape: (b, c, l)

        #
        x = self.unpatch_wave(x_wave, self.ac)
        x = x[:, :, 0 : T_2048, :]
        y_wave = self.bases_wave.decode(x, audio.shape[-1])

        output = (y_2048 + y_wave) / 2

        return output

    def pad_tensor(self, x: Tensor) -> tuple[Tensor, int]:
        r"""Pad a spectrum that can be evenly divided by downsample_ratio.

        Args:
            x: E.g., (b, c, t=201, f)
        
        Outpus:
            output: E.g., (b, c, t=204, f)
        """

        # Pad last frames, e.g., 201 -> 204
        pad_t = -x.shape[2] % self.patch_size[0]  # Equals to p - (T % p)
        x = F.pad(x, pad=(0, 0, 0, 0, 0, pad_t))

        return x

    def pad_tensor2(self, x):
        pad_t = -x.shape[2] % self.patch_size[0]  # Equals to p - (T % p)
        x = F.pad(x, pad=(0, 0, 0, pad_t))

        return x


class Patch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=kernel_size)

    def __call__(self, x):
        x = rearrange(x, 'b c t f i -> b (c i) t f')
        x = self.conv(x)
        return x


class UnPatch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=kernel_size)

    def __call__(self, x, audio_channels):
        x = self.conv(x)
        x = rearrange(x, 'b (c i) t f -> b c t f i', c=audio_channels)
        
        return x


class Patch2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=kernel_size)

    def __call__(self, x):
        x = self.conv(x)
        return x


class UnPatch2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=kernel_size)

    def __call__(self, x, audio_channels):
        x = self.conv(x)
        return x


class BasesWave(nn.Module):
    def __init__(self, n_window, hop_length):
        super().__init__()
        self.n_window = n_window
        self.hop_length = hop_length
        self.register_buffer(name="window", tensor=torch.hann_window(n_window))

    def encode(self, x):
        B, C, L = x.shape
        x = rearrange(x, 'b c l -> (b c) l')
        x = bases_wave_transform(x, self.n_window, self.hop_length, self.window)
        x = rearrange(x, '(b c) t f -> b c t f', b=B)
        return x

    def decode(self, x, length):
        B, C, T, F_ = x.shape
        x = rearrange(x, 'b c t f -> (b c) t f')
        x = bases_wave_inverse_transform(x, self.n_window, self.hop_length, self.window)
        x = rearrange(x, '(b c) l -> b c l', b=B)
        return x[:, :, 0 : length]