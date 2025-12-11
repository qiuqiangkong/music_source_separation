from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
import time

from mss.models.attention import Block
from mss.models.bandsplit import BandSplit
from mss.models.fourier import Fourier
from mss.models.rope import RoPE

from mss.models2.gabor_transform import GaborTransform


class BSRoformer26a(nn.Module): 
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

        self.patch_size = patch_size

        self.fourier = GaborTransform(
            n_ffts=[n_fft],
            hop_lengths=[hop_length],
            r=1,
        )

        # Band split
        self.bandsplit = BandSplit(
            sr=sample_rate, 
            n_fft=n_fft, 
            n_bands=n_bands,
            in_channels=audio_channels * 2,  # real + imag
            out_channels=band_dim
        )

        self.patch = nn.Conv2d(band_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.unpatch = nn.ConvTranspose2d(dim, band_dim, kernel_size=patch_size, stride=patch_size)

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

        features = self.fourier.encode(audio)
        complex_sp = features[0]
        
        x = torch.view_as_real(complex_sp)  # shape: (b, c, t, f, 2)
        x = rearrange(x, 'b c t f k -> b (c k) t f')  # shape: (b, d, t, f)
        T0 = x.shape[2]

        # 1.2 Pad stft
        x = self.pad_tensor(x)  # x: (b, d, t, f)

        # 1.3 Convert STFT to mel scale
        x = self.bandsplit.transform(x)  # shape: (b, d, t, f)

        # 1.4 Patchify
        x = self.patch(x)  # shape: (b, d, t, f)
        B = x.shape[0]
        T1 = x.shape[2]

        # --- 2. Transformer along time and frequency axes ---
        for t_block, f_block in zip(self.t_blocks, self.f_blocks):

            x = rearrange(x, 'b d t f -> (b f) t d')
            x = t_block(x, rope=self.rope, pos=None)  # shape: (b*f, t, d)

            x = rearrange(x, '(b f) t d -> (b t) f d', b=B)
            x = f_block(x, rope=self.rope, pos=None)  # shape: (b*t, f, d)

            x = rearrange(x, '(b t) f d -> b d t f', b=B)  # shape: (b, d, t, f)

        # --- 3. Decode ---
        # 3.1 Unpatchify
        x = self.unpatch(x)  # shape: (b, d, t, f)

        # 3.2 Convert mel scale STFT to original STFT
        x = self.bandsplit.inverse_transform(x)  # shape: (b, d, t, f)

        # Unpad
        x = x[:, :, 0 : T0, :]
        
        # 3.3 Get complex mask
        x = rearrange(x, 'b (c k) t f -> b c t f k', k=2).contiguous()
        mask = torch.view_as_complex(x)  # shape: (b, c, t, f)

        # 3.5 Calculate stft of separated audio
        sep_stft = mask * complex_sp  # shape: (b, c, t, f)

        # 3.6 ISTFT
        output = self.fourier.decode([sep_stft], audio.shape[-1])
        # from IPython import embed; embed(using=False); os._exit(0)
        # output = self.istft(sep_stft)  # shape: (b, c, l)

        return output
    '''
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

        torch.cuda.synchronize()
        t1 = time.time()
        features = self.fourier.encode(audio)
        complex_sp = features[0]
        torch.cuda.synchronize()
        print("a1", time.time() - t1)
        torch.cuda.synchronize()
        
        x = torch.view_as_real(complex_sp)  # shape: (b, c, t, f, 2)
        x = rearrange(x, 'b c t f k -> b (c k) t f')  # shape: (b, d, t, f)
        T0 = x.shape[2]

        # 1.2 Pad stft
        x = self.pad_tensor(x)  # x: (b, d, t, f)

        torch.cuda.synchronize()
        t1 = time.time()

        # 1.3 Convert STFT to mel scale
        x = self.bandsplit.transform(x)  # shape: (b, d, t, f)

        torch.cuda.synchronize()
        print("a2", time.time() - t1)
        torch.cuda.synchronize()

        # 1.4 Patchify
        x = self.patch(x)  # shape: (b, d, t, f)
        B = x.shape[0]
        T1 = x.shape[2]

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

        # --- 3. Decode ---
        # 3.1 Unpatchify
        x = self.unpatch(x)  # shape: (b, d, t, f)

        torch.cuda.synchronize()
        t1 = time.time()

        # 3.2 Convert mel scale STFT to original STFT
        x = self.bandsplit.inverse_transform(x)  # shape: (b, d, t, f)

        torch.cuda.synchronize()
        print("a4", time.time() - t1)
        torch.cuda.synchronize()

        # Unpad
        x = x[:, :, 0 : T0, :]
        
        # 3.3 Get complex mask
        x = rearrange(x, 'b (c k) t f -> b c t f k', k=2).contiguous()
        mask = torch.view_as_complex(x)  # shape: (b, c, t, f)

        # 3.5 Calculate stft of separated audio
        sep_stft = mask * complex_sp  # shape: (b, c, t, f)

        torch.cuda.synchronize()
        t1 = time.time()

        # 3.6 ISTFT
        output = self.fourier.decode([sep_stft], audio.shape[-1])
        # from IPython import embed; embed(using=False); os._exit(0)
        # output = self.istft(sep_stft)  # shape: (b, c, l)

        torch.cuda.synchronize()
        print("a5", time.time() - t1)
        torch.cuda.synchronize()

        return output
    '''
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

        return x