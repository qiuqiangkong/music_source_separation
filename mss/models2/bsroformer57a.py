from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from mss.models.attention import Block
from mss.models2.bandsplit42a import BandSplit
from mss.models.fourier import Fourier
from mss.models.rope import RoPE
import time



class BSRoformer57a(Fourier):
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
        
        self.encoder = Encoder(
            in_channels=audio_channels * 2, 
            dims=[32, 96, 384], 
            kernel_sizes=[(1, 1), (1, 4), (4, 4)],
            n_layers=[1, 1, 6], 
            
        )

        self.decoder = Decoder(
            dims=[32, 96, 384], 
            kernel_sizes=[(1, 1), (1, 4), (4, 4)],
            n_layers=[1, 1, 6], 
            out_channels=audio_channels * 2, 
        )

        # self.encoder = Encoder(
        #     in_channels=audio_channels * 2, 
        #     dims=[32, 96, 96], 
        #     kernel_sizes=[(1, 1), (4, 4), (1, 1)],
        #     n_layers=[1, 3, 3], 
            
        # )

        # self.decoder = Decoder(
        #     dims=[32, 96, 96], 
        #     kernel_sizes=[(1, 1), (4, 4), (1, 1)],
        #     n_layers=[1, 3, 3], 
        #     out_channels=audio_channels * 2, 
        # )

        # RoPE
        self.rope = RoPE(head_dim=dim // n_heads, max_len=rope_len)

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

        # 1.2 Pad stft
        x = self.pad_tensor(complex_sp)  # x: (b, d, t, f)

        x = torch.view_as_real(x)  # shape: (b, c, t, f, 2)
        x = rearrange(x, 'b c t f k -> b (c k) t f')

        #
        enc1, enc2, enc3 = self.encoder(x, self.rope)

        x = self.decoder(enc3, enc2, enc1, self.rope)

        # 3.3 Get complex mask
        x = rearrange(x, 'b (c k) t f -> b c t f k', k=2).contiguous()
        x = torch.view_as_complex(x)  # shape: (b, c, t, f)

        # Unpad
        mask = F.pad(x[:, :, 0 : T0, :], pad=(0, 1))

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
        x = F.pad(x, pad=(0, 0, 0, pad_t))
        x = x[..., 0 : -1]

        return x


class Patch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=kernel_size)

    def __call__(self, x):
        x = self.conv(x)
        return x


class UnPatch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=kernel_size)

    def __call__(self, x):
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, dims, kernel_sizes, n_layers):
        super().__init__()
        
        self.patch1 = nn.Conv2d(in_channels, dims[0], kernel_size=kernel_sizes[0], stride=kernel_sizes[0])
        self.blocks1 = nn.ModuleList(Block(dims[0], dims[0]//32) for _ in range(n_layers[0]))

        self.patch2 = nn.Conv2d(dims[0], dims[1], kernel_size=kernel_sizes[1], stride=kernel_sizes[1])
        self.blocks2 = nn.ModuleList(Block(dims[1], dims[1]//32) for _ in range(n_layers[1]))

        self.patch3 = nn.Conv2d(dims[1], dims[2], kernel_size=kernel_sizes[2], stride=kernel_sizes[2])
        self.blocks3t = nn.ModuleList(Block(dims[2], dims[2]//32) for _ in range(n_layers[2]))
        self.blocks3f = nn.ModuleList(Block(dims[2], dims[2]//32) for _ in range(n_layers[2]))

    def __call__(self, x, rope):
        
        B = x.shape[0]

        # B = x.shape[0]
        # x0 = x
        # while True:
        #     t1 = time.time()
        #     x = rearrange(x0, 'b d t (f1 f2) -> (b t f1) f2 d', f1=1)
        #     for block in self.blocks1:
        #         x = block(x, rope=rope, pos=None)
        #     x = rearrange(x, '(b t f1) f2 d -> b d t (f1 f2)', b=B, f1=1)
        #     print(time.time() - t1)

        #
        # x = self.patch1(x)
        # x = rearrange(x, 'b d t (f1 f2) -> (b t f1) f2 d', f1=4)
        # for block in self.blocks1:
        #     x = block(x, rope=rope, pos=None)
        # x = rearrange(x, '(b t f1) f2 d -> b d t (f1 f2)', b=B, f1=4)
        # enc1 = x
        # from IPython import embed; embed(using=False); os._exit(0)

        x = self.patch1(x)
        x = rearrange(x, 'b d t f -> (b t) f d')
        for block in self.blocks1:
            x = block(x, rope=rope, pos=None)
        x = rearrange(x, '(b t) f d -> b d t f', b=B)
        enc1 = x

        # 
        x = self.patch2(x)
        x = rearrange(x, 'b d t f -> (b t) f d')
        for block in self.blocks2:
            x = block(x, rope=rope, pos=None)
        x = rearrange(x, '(b t) f d -> b d t f', b=B)
        enc2 = x

        # 
        x = self.patch3(x)
        
        for t_block, f_block in zip(self.blocks3t, self.blocks3f):
            x = rearrange(x, 'b d t f -> (b f) t d')
            x = t_block(x, rope=rope, pos=None)  # shape: (b*f, t, d)

            x = rearrange(x, '(b f) t d -> (b t) f d', b=B)
            x = f_block(x, rope=rope, pos=None)  # shape: (b*t, f, d)

            x = rearrange(x, '(b t) f d -> b d t f', b=B)  # shape: (b, d, t, f)
        
        enc3 = x

        return enc1, enc2, enc3


class Decoder(nn.Module):
    def __init__(self, dims, kernel_sizes, n_layers, out_channels):
        super().__init__()
        
        self.blocks3t = nn.ModuleList(Block(dims[2], dims[2]//32) for _ in range(n_layers[2]))
        self.blocks3f = nn.ModuleList(Block(dims[2], dims[2]//32) for _ in range(n_layers[2]))
        self.unpatch3 = nn.ConvTranspose2d(dims[2], dims[1], kernel_size=kernel_sizes[2], stride=kernel_sizes[2])

        self.blocks2 = nn.ModuleList(Block(dims[1], dims[1]//32) for _ in range(n_layers[1]))
        self.unpatch2 = nn.ConvTranspose2d(dims[1], dims[0], kernel_size=kernel_sizes[1], stride=kernel_sizes[1])

        self.blocks1 = nn.ModuleList(Block(dims[0], dims[0]//32) for _ in range(n_layers[0]))
        self.unpatch1 = nn.ConvTranspose2d(dims[0], out_channels, kernel_size=kernel_sizes[0], stride=kernel_sizes[0])

    def __call__(self, enc3, enc2, enc1, rope):
        
        B = enc3.shape[0]

        #
        x = enc3

        for t_block, f_block in zip(self.blocks3t, self.blocks3f):
            x = rearrange(x, 'b d t f -> (b f) t d')
            x = t_block(x, rope=rope, pos=None)  # shape: (b*f, t, d)

            x = rearrange(x, '(b f) t d -> (b t) f d', b=B)
            x = f_block(x, rope=rope, pos=None)  # shape: (b*t, f, d)

            x = rearrange(x, '(b t) f d -> b d t f', b=B)  # shape: (b, d, t, f)

        x = self.unpatch3(x)

        #
        x += enc2
        x = rearrange(x, 'b d t f -> (b t) f d')
        for block in self.blocks2:
            x = block(x, rope=rope, pos=None)
        x = rearrange(x, '(b t) f d -> b d t f', b=B)
        x = self.unpatch2(x)

        #
        x += enc1
        x = rearrange(x, 'b d t f -> (b t) f d')
        for block in self.blocks1:
            x = block(x, rope=rope, pos=None)
        x = rearrange(x, '(b t) f d -> b d t f', b=B)
        x = self.unpatch1(x)

        return x