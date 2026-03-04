from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from mss.models.attention import Block
from mss.models2.bandsplit42a import BandSplit
from mss.models2.bandsplit_linear import BandSplitLinear
from mss.models.fourier import Fourier
from mss.models.rope import RoPE
import time



class BSRoformer66a(nn.Module):
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
        # super().__init__(
        #     n_fft=n_fft, 
        #     hop_length=hop_length, 
        #     return_complex=True, 
        #     normalized=True
        # )
        self.ac = audio_channels
        self.patch_size = patch_size

        self.fourier_128 = Fourier(n_fft=128, hop_length=30, return_complex=True, normalized=True)
        self.fourier_512 = Fourier(n_fft=512, hop_length=120, return_complex=True, normalized=True)
        self.fourier_2048 = Fourier(n_fft=2048, hop_length=480, return_complex=True, normalized=True)

        # Band split
        self.bandsplit = BandSplit(
            sr=sample_rate, 
            n_fft=n_fft, 
            n_bands=n_bands,
            in_channels=2,  # real + imag
            out_channels=band_dim
        )

        self.patch_128 = Patch(4, 384, (64, 1))
        self.patch_512 = Patch(4, 384, (16, 4))
        self.patch_2048 = Patch(band_dim * audio_channels, dim, (4, 4))

        self.pre_fc = nn.Linear(384*3, 384)
        self.post_fc = nn.Linear(384, 384*3)

        # kernel_size = (4, 4)
        kernel_size = patch_size
        
        self.unpatch_128 = UnPatch(384, 4, (64, 1))
        self.unpatch_512 = UnPatch(384, 4, (16, 4))
        self.unpatch_2048 = UnPatch(dim, band_dim * audio_channels, kernel_size)

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

        sp_128 = self.fourier_128.stft(audio)
        T_128 = sp_128.shape[2]
        x = torch.view_as_real(sp_128)  # shape: (b, c, t, f, 2)
        x = self.pad_tensor(x, 64)  # x: (b, d, t, f)
        x = x[:, :, :, 0 : -1, :]
        x = self.patch_128(x)
        x_128 = x

        sp_512 = self.fourier_512.stft(audio)
        T_512 = sp_512.shape[2]
        x = torch.view_as_real(sp_512)  # shape: (b, c, t, f, 2)
        x = self.pad_tensor(x, 16)  # x: (b, d, t, f)
        x = x[:, :, :, 0 : -1, :]
        x = self.patch_512(x)
        x_512 = x
        
        
        sp_2048 = self.fourier_2048.stft(audio)
        T_2048 = sp_2048.shape[2]
        x = torch.view_as_real(sp_2048)  # shape: (b, c, t, f, 2)
        x = self.pad_tensor(x, 4)  # x: (b, d, t, f)
        x = self.bandsplit.transform(x)  # shape: (b, c, t, f, o)
        x = self.patch_2048(x)
        x_2048 = x

        x = torch.cat([x_128, x_512, x_2048], dim=1)
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
        x_128, x_512, x_2048 = x.chunk(chunks=3, dim=1)

        #
        x = self.unpatch_2048(x_2048, self.ac)
        x = self.bandsplit.inverse_transform(x)  # shape: (b, c, t, f, k)
        x = x[:, :, 0 : T_2048, :, :]
        mask = torch.view_as_complex(x)  # shape: (b, c, t, f)
        y_2048 = mask * sp_2048  # shape: (b, c, t, f)
        y_2048 = self.fourier_2048.istft(y_2048)  # shape: (b, c, l)

        x = self.unpatch_512(x_512, self.ac)
        x = F.pad(x, pad=(0, 0, 0, 1))
        x = x[:, :, 0 : T_512, :, :]
        mask = torch.view_as_complex(x)  # shape: (b, c, t, f)
        y_512 = mask * sp_512  # shape: (b, c, t, f)
        y_512 = self.fourier_512.istft(y_512)  # shape: (b, c, l)

        x = self.unpatch_128(x_128, self.ac)
        x = F.pad(x, pad=(0, 0, 0, 1))
        x = x[:, :, 0 : T_128, :, :]
        mask = torch.view_as_complex(x)  # shape: (b, c, t, f)
        y_128 = mask * sp_128  # shape: (b, c, t, f)
        y_128 = self.fourier_128.istft(y_128)  # shape: (b, c, l)

        output = y_128 + y_512 + y_2048

        return output

    def pad_tensor(self, x: Tensor, patch_size_t: int) -> tuple[Tensor, int]:
        r"""Pad a spectrum that can be evenly divided by downsample_ratio.

        Args:
            x: E.g., (b, c, t=201, f)
        
        Outpus:
            output: E.g., (b, c, t=204, f)
        """

        # Pad last frames, e.g., 201 -> 204
        pad_t = -x.shape[2] % patch_size_t  # Equals to p - (T % p)
        x = F.pad(x, pad=(0, 0, 0, 0, 0, pad_t))

        return x

'''
class BSRoformer42a(Fourier):
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

        super().__init__(
            n_fft=n_fft, 
            hop_length=hop_length, 
            return_complex=True, 
            normalized=True
        )

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

        kernel_size = (4, 4)
        self.patch = Patch(band_dim * audio_channels, dim, kernel_size)
        self.unpatch = UnPatch(dim, band_dim * audio_channels, kernel_size)

        # self.patch_f = nn.Conv1d(in_channels=band_dim, out_channels=dim, kernel_size=4, stride=4)
        # self.patch_t = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=4, stride=4)
        # # self.patch_c = nn.Conv1d(in_channels=band_dim, out_channels=dim, kernel_size=audio_channels, stride=audio_channels)
        # self.patch_c = nn.Linear(dim * audio_channels, dim)

        # self.patch = nn.Conv2d(band_dim, dim, kernel_size=patch_size, stride=patch_size)
        # self.unpatch = nn.ConvTranspose2d(dim, band_dim, kernel_size=patch_size, stride=patch_size)

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
        complex_sp = self.stft(audio)  # shape: (b, c, t, f)
        T0 = complex_sp.shape[2]

        x = torch.view_as_real(complex_sp)  # shape: (b, c, t, f, 2)

        # 1.2 Pad stft
        x = self.pad_tensor(x)  # x: (b, d, t, f)

        torch.cuda.synchronize()
        t1 = time.time()
        # 1.3 Convert STFT to mel scale
        x = self.bandsplit.transform(x)  # shape: (b, c, t, f, o)
        torch.cuda.synchronize()
        print("a1", time.time() - t1)
        torch.cuda.synchronize()

        x = self.patch(x)

        B = x.shape[0]
        # T1 = x.shape[2]

        torch.cuda.synchronize()
        t1 = time.time()
        # --- 2. Transformer along time and frequency axes ---
        for t_block, f_block in zip(self.t_blocks, self.f_blocks):

            x = rearrange(x, 'b d t f -> (b f) t d')
            x = t_block(x, rope=self.rope, pos=None)  # shape: (b*f, t, d)

            x = rearrange(x, '(b f) t d -> (b t) f d', b=B)
            x = f_block(x, rope=self.rope, pos=None)  # shape: (b*t, f, d)

            x = rearrange(x, '(b t) f d -> b d t f', b=B)  # shape: (b, d, t, f)

        torch.cuda.synchronize()
        print("a2", time.time() - t1)
        torch.cuda.synchronize()

        # --- 3. Decode ---
        # 3.1 Unpatchify
        x = self.unpatch(x, self.ac)

        torch.cuda.synchronize()
        t1 = time.time()

        # 3.2 Convert mel scale STFT to original STFT
        x = self.bandsplit.inverse_transform(x)  # shape: (b, c, t, f, k)

        torch.cuda.synchronize()
        print("a3", time.time() - t1)
        torch.cuda.synchronize()

        # Unpad
        x = x[:, :, 0 : T0, :, :]
        
        # 3.3 Get complex mask
        mask = torch.view_as_complex(x)  # shape: (b, c, t, f)

        # 3.5 Calculate stft of separated audio
        sep_stft = mask * complex_sp  # shape: (b, c, t, f)

        # 3.6 ISTFT
        output = self.istft(sep_stft)  # shape: (b, c, l)

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
'''

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