from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from mss.models.attention import Block
from mss.models.bandsplit import BandSplit
from mss.models.fourier import Fourier
from mss.models.rope import RoPE


class BSRoformer11a(Fourier):
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

        self.patch_size = patch_size
        self.dim = dim
        self.band_dim = band_dim

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

        token_num = 8
        self.wavfeat = WavFeat(n_fft, hop_length, dim * token_num)
        self.wavpatch = nn.Conv2d(dim, dim, kernel_size=(patch_size[0], 1), stride=(patch_size[0], 1))
        self.wavunpatch = nn.ConvTranspose2d(dim, dim, kernel_size=(patch_size[0], 1), stride=(patch_size[0], 1))

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
        F1 = x.shape[3]

        x_wav = self.wavfeat.encode(audio)
        C = audio.shape[1]
        x_wav = rearrange(x_wav, 'b c t (f d) -> b d t (c f)', d=self.dim)
        x_wav = self.pad_tensor(x_wav)
        x_wav = self.wavpatch(x_wav)
        
        x = torch.cat([x, x_wav], dim=-1)

        # --- 2. Transformer along time and frequency axes ---
        for t_block, f_block in zip(self.t_blocks, self.f_blocks):

            x = rearrange(x, 'b d t f -> (b f) t d')
            x = t_block(x, rope=self.rope, pos=None)  # shape: (b*f, t, d)

            x = rearrange(x, '(b f) t d -> (b t) f d', b=B)
            x = f_block(x, rope=self.rope, pos=None)  # shape: (b*t, f, d)

            x = rearrange(x, '(b t) f d -> b d t f', b=B)  # shape: (b, d, t, f)

        x_mel = x[:, :, :, 0 : F1]
        x_wav = x[:, :, :, F1:]

        # --- 3. Decode ---
        # 3.1 Unpatchify
        x = self.unpatch(x_mel)  # shape: (b, d, t, f)

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
        output1 = self.istft(sep_stft)  # shape: (b, c, l)

        x_wav = self.wavunpatch(x_wav)
        x_wav = x_wav[:, :, 0 : T0, :]
        x_wav = rearrange(x_wav, 'b d t (c f) -> b c t (f d)', c=C)
        output2 = self.wavfeat.decode(x_wav, audio.shape[-1])

        output = output1 + output2

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

        return x


class WavFeat(nn.Module):
    def __init__(self, n_fft, hop_length, dim):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        window = torch.hann_window(n_fft)
        self.register_buffer("window", window)

        M = dim
        N = self.n_fft
        self.W_enc = nn.Parameter(0.0001 * torch.randn(M, N))
        self.W_dec = nn.Parameter(0.0001 * torch.randn(N, M))

    def encode(self, x: Tensor) -> Tensor:
        r"""

        b: batch_size
        c: num_channels
        l: segment_samples
        t: num_frames
        f: freq_bins
        n: frame_samples

        Args:
            x: (b, c, l)

        Returns: 
            out: (b, c, t, f)
        """

        x = F.pad(x, (self.n_fft // 2, self.n_fft // 2), mode="reflect")
        x = x.unfold(dimension=-1, size=self.n_fft, step=self.hop_length).contiguous()  # (b, t, n)
        x *= self.window
        out = x @ self.W_enc.T
        return out

    def decode(self, x: Tensor, length: int | None) -> Tensor:
        r"""

        b: batch_size
        c: num_channels
        l: segment_samples
        t: num_frames
        f: freq_bins

        Args:
            x: (b, c, t, f)

        Returns:
            out: (b, c, l)
        """

        # Inverse transform
        x = x @ self.W_dec.T

        # Overlap-add
        out = fold(
            x=rearrange(x, 'b c t n -> (b c) n t'), 
            hop_length=self.hop_length, 
            window=self.window
        )  # (b*c, l)
        out = rearrange(out, '(b c) l -> b c l', b=x.shape[0])
        
        # Remove padding
        out = out[..., self.n_fft // 2 :]
        
        if length is not None:
            out = out[..., 0 : length]

        return out


def fold(x: Tensor, hop_length: int, window: Tensor | None):
    r"""

    b: batch_size
    t: num_frames
    n: frame_samples
    l: segment_samples

    Args:
        x: (b, n, t)

    Returns:
        x: (b, l)
    """

    frame_length, num_frames = x.shape[-2:]  # (t, n)
    L = frame_length + (num_frames - 1) * hop_length

    # Overlap-add
    x = F.fold(
        input=x,  # (b, n, t)
        output_size=(1, L),
        kernel_size=(1, frame_length),
        stride=(1, hop_length)
    )  # (b, c, 1, l)
    out = x.squeeze(dim=[1, 2])  # (b, l)

    # Divide overlap-add window
    if window is not None:
        win_norm = F.fold(
            window[None, :, None].repeat(1, 1, num_frames),  # (1, n, t),
            output_size=(1, L),
            kernel_size=(1, frame_length),
            stride=(1, hop_length)
        )  # (1, 1, 1, L)
        win_norm = win_norm.squeeze(dim=[0, 1, 2])  # (l,)

        out /= torch.clamp(win_norm, 1e-8)  # (b, l)

    return out
