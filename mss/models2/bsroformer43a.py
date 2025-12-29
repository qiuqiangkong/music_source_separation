from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from mss.models2.attention43a import Block
from mss.models2.bandsplit42a import BandSplit
from mss.models.fourier import Fourier
from mss.models.rope import RoPE
from mss.models2.gabor_transform import GaborTransform
import time


class BSRoformer43a(Fourier):
    def __init__(
        self,
        audio_channels=2,
        sample_rate=48000,
        n_ffts=[2048],
        hop_lengths=[512],
        oversampling_factors=[16],
        n_bands=[256],
        band_dims=[64],
        patch_sizes=[[4, 4]],
        n_layers=12,
        n_heads=12,
        dim=768,
        rope_len=8192,
        **kwargs
    ) -> None:

        super().__init__()

        self.ac = audio_channels
        self.n_windows = len(n_ffts)
        self.hop_lengths = hop_lengths
        self.patch_sizes = patch_sizes

        self.gabor = GaborTransform(
            n_ffts=n_ffts,
            hop_lengths=hop_lengths,
            oversampling_factors=oversampling_factors
        )

        self.bandsplits = nn.ModuleList([])
        self.patches = nn.ModuleList([])
        self.unpatches = nn.ModuleList([])
        
        for i in range(self.n_windows):
            self.bandsplits.append(BandSplit(
                sr=sample_rate, 
                n_fft=n_ffts[i] * oversampling_factors[i], 
                n_bands=n_bands[i],
                in_channels=2,  # real + imag
                out_channels=band_dims[i]
            ))
            self.patches.append(Patch(band_dims[i] * audio_channels, dim, patch_sizes[i]))
            self.unpatches.append(UnPatch(dim, band_dims[i] * audio_channels, patch_sizes[i]))

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

        B, C, L = audio.shape
        audio = self.pad_tensor(audio)

        features = self.gabor.transform(audio)  # list of (b, c, t, f)
        features = [feat[:, :, 0 : -1, :] for feat in features]

        emb = 0
        for i in range(self.n_windows):
            x = torch.view_as_real(features[i])  # shape: (b, c, t, f, 2)
            x = self.bandsplits[i].transform(x)  # (b, c, t, s, o)
            x = self.patches[i](x)  # (b, d, t, f)
            emb += x

        # --- 2. Transformer along time and frequency axes ---
        for t_block, f_block in zip(self.t_blocks, self.f_blocks):

            x = rearrange(x, 'b d t f -> (b f) t d')
            x = t_block(x, rope=self.rope, pos=None)  # shape: (b*f, t, d)

            x = rearrange(x, '(b f) t d -> (b t) f d', b=B)
            x = f_block(x, rope=self.rope, pos=None)  # shape: (b*t, f, d)

            x = rearrange(x, '(b t) f d -> b d t f', b=B)  # shape: (b, d, t, f)

        masks = []
        for i in range(self.n_windows):
            y = self.unpatches[i](x, self.ac)  # (b, c, t, s, o)
            y = self.bandsplits[i].inverse_transform(y)  # (b, c, t, f, 2)
            y = torch.view_as_complex(y)
            masks.append(y)
            
        out = [feat * mask for feat, mask in zip(features, masks)]
        out = self.gabor.inverse_transform(out, audio.shape[-1])
        out = out[:, :, 0 : L]

        return out

    def pad_tensor(self, x: Tensor) -> tuple[Tensor, int]:
        r"""Pad a spectrum that can be evenly divided by downsample_ratio.

        Args:
            x: E.g., (b, c, t=201, f)
        
        Outpus:
            output: E.g., (b, c, t=204, f)
        """

        max_hop = max([hop * ps[0] for hop, ps in zip(self.hop_lengths, self.patch_sizes)])
        pad_t = -x.shape[2] % max_hop + max_hop
        x = F.pad(x, pad=(0, pad_t))
        return x
'''

class BSRoformer43a(Fourier):
    def __init__(
        self,
        audio_channels=2,
        sample_rate=48000,
        n_ffts=[2048],
        hop_lengths=[512],
        oversampling_factors=[16],
        n_bands=[256],
        band_dims=[64],
        patch_sizes=[[4, 4]],
        n_layers=12,
        n_heads=12,
        dim=768,
        rope_len=8192,
        **kwargs
    ) -> None:

        super().__init__()

        self.ac = audio_channels
        self.n_windows = len(n_ffts)
        self.hop_lengths = hop_lengths
        self.patch_sizes = patch_sizes

        self.gabor = GaborTransform(
            n_ffts=n_ffts,
            hop_lengths=hop_lengths,
            oversampling_factors=oversampling_factors
        )

        self.bandsplits = nn.ModuleList([])
        self.patches = nn.ModuleList([])
        self.unpatches = nn.ModuleList([])
        
        for i in range(self.n_windows):
            self.bandsplits.append(BandSplit(
                sr=sample_rate, 
                n_fft=n_ffts[i] * oversampling_factors[i], 
                n_bands=n_bands[i],
                in_channels=2,  # real + imag
                out_channels=band_dims[i]
            ))
            self.patches.append(Patch(band_dims[i] * audio_channels, dim, patch_sizes[i]))
            self.unpatches.append(UnPatch(dim, band_dims[i] * audio_channels, patch_sizes[i]))

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

        B, C, L = audio.shape
        audio = self.pad_tensor(audio)

        torch.cuda.synchronize()
        t1 = time.time()

        features = self.gabor.transform(audio)  # list of (b, c, t, f)
        features = [feat[:, :, 0 : -1, :] for feat in features]

        torch.cuda.synchronize()
        print("a1", time.time() - t1)
        torch.cuda.synchronize()

        torch.cuda.synchronize()
        t1 = time.time()

        emb = 0
        for i in range(self.n_windows):
            x = torch.view_as_real(features[i])  # shape: (b, c, t, f, 2)
            x = self.bandsplits[i].transform(x)  # (b, c, t, s, o)
            x = self.patches[i](x)  # (b, d, t, f)
            emb += x

        torch.cuda.synchronize()
        print("a2", time.time() - t1)
        torch.cuda.synchronize()

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
        print("a3", time.time() - t1)
        torch.cuda.synchronize()

        torch.cuda.synchronize()
        t1 = time.time()

        masks = []
        for i in range(self.n_windows):
            y = self.unpatches[i](x, self.ac)  # (b, c, t, s, o)
            y = self.bandsplits[i].inverse_transform(y)  # (b, c, t, f, 2)
            y = torch.view_as_complex(y)
            masks.append(y)
        
        torch.cuda.synchronize()
        print("a4", time.time() - t1)
        torch.cuda.synchronize()

        torch.cuda.synchronize()
        t1 = time.time()

        out = [feat * mask for feat, mask in zip(features, masks)]
        out = self.gabor.inverse_transform(out, audio.shape[-1])
        out = out[:, :, 0 : L]

        torch.cuda.synchronize()
        print("a5", time.time() - t1)
        torch.cuda.synchronize()

        return out

    def pad_tensor(self, x: Tensor) -> tuple[Tensor, int]:
        r"""Pad a spectrum that can be evenly divided by downsample_ratio.

        Args:
            x: E.g., (b, c, t=201, f)
        
        Outpus:
            output: E.g., (b, c, t=204, f)
        """

        max_hop = max([hop * ps[0] for hop, ps in zip(self.hop_lengths, self.patch_sizes)])
        pad_t = -x.shape[2] % max_hop + max_hop
        x = F.pad(x, pad=(0, pad_t))
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