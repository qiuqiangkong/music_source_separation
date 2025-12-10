from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
import librosa
import time

from mss.models.attention import Block
from mss.models.bandsplit import BandSplit
from mss.models.fourier import Fourier
from mss.models.rope import RoPE
import numpy as np
import math
import pickle
from pathlib import Path



class BSRoformer20b(Fourier):
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

        # Band split
        self.bandsplit = BandSplit(
            sr=sample_rate, 
            n_fft=n_fft * 16, 
            n_bands=n_bands,
            in_channels=audio_channels * 2,  # real + imag
            out_channels=band_dim
        )

        self.fourier = FractionalSTFT(n_fft, hop_length)

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

        # --- 1. Encode ---
        # 1.1 Complex spectrum
        complex_sp = self.fourier.encode(audio)  # (b, c, t, f1*f2)

        # from IPython import embed; embed(using=False); os._exit(0)
        # import matplotlib.pyplot as plt
        # plt.matshow(np.log(complex_sp.abs().cpu().numpy()[0, 0]).T, origin='lower', aspect='auto', cmap='jet')
        # plt.savefig("_zz.pdf")
        
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
        output = self.fourier.decode(sep_stft, audio.shape[-1])  # shape: (b, c, l)

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


def hz_to_mel(f):
    return 2595 * np.log10(1 + f / 700)


def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)


class FractionalSTFT(nn.Module):
    def __init__(self, n_fft: int, hop_length: int, recompute=False):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        window = torch.hann_window(n_fft)

        N = n_fft
        n = torch.arange(0, N)

        # k1 = torch.arange(0, N // 2 + 1)
        k1 = torch.arange(0, N // 2 + 1e-6, 1/16)
        k1_flip = N - torch.flip(k1, dims=[0])[1 : -1]
        k2 = torch.cat([k1, k1_flip], dim=0)

        W_enc = torch.exp(-1.j * 2 * math.pi / N * torch.outer(k1, n)) / math.sqrt(N)
        W = torch.exp(-1.j * 2 * math.pi / N * torch.outer(k2, n)) / math.sqrt(N)

        W_dec_path = "_W_dec_20b.pkl"
        if Path(W_dec_path).is_file():
            W_dec = pickle.load(open(W_dec_path, "rb"))
            print(f"Load from {W_dec_path}")
        else:
            t1 = time.time()
            # W_dec = torch.linalg.pinv(W)
            W_dec = torch.linalg.pinv(W.to("cuda")).cpu()
            pickle.dump(W_dec, open(W_dec_path, "wb"))
            print(f"Write out to {W_dec_path}")
            print("time: {:.2f} s".format(time.time() - t1))
        
        self.register_buffer("window", window)
        self.register_buffer("W_enc", W_enc)
        self.register_buffer("W_dec", W_dec)

        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots(5, 1, sharex=True)
        # axs[0].plot(W_dec[:, 0].real)
        # axs[1].plot(W_dec[:, 10].real)
        # axs[2].plot(W_dec[:, 100].real)
        # axs[3].plot(W_dec[:, 1000].real)
        # plt.savefig("_zz.pdf")
        # from IPython import embed; embed(using=False); os._exit(0) 

    def init_melbanks(self) -> tuple[np.ndarray, np.ndarray]:
        r"""Initialize mel bins from librosa.

        f: stft bins
        k: mel bins

        Args:
            None

        Returns:
            melbanks: (k, f)
            ola_window: (f,)
        """

        melbanks = librosa.filters.mel(
            sr=self.sr, 
            n_fft=self.n_fft * 16, 
            n_mels=self.n_bands - 2, 
            norm=None
        )  # shape: (k, f)

        F = self.n_fft // 2 + 1
        
        # The zeroth bank, e.g., [1., 0.66, 0.32, 0, ..., 0.]
        melbank_0 = np.zeros(F)
        idx = np.argmax(melbanks[0])
        melbank_0[0 : idx] = 1. - melbanks[0, 0 : idx]  # (f,)

        # The last bank, e.g., [0., ..., 0., 0.18, 0.87, 1.]
        melbank_last = np.zeros(F)
        idx = np.argmax(melbanks[-1])
        melbank_last[idx :] = 1. - melbanks[-1, idx :]  # (f,)

        # Concatenate
        melbanks = np.concatenate(
            [melbank_0[None, :], melbanks, melbank_last[None, :]], axis=0
        )  # (n_mels, f)

        # Calculate overlap-add window
        ola_window = np.sum(melbanks, axis=0)  # overlap add window
        assert ola_window.max() >= 0.5

        return melbanks, ola_window

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
        out = x.to(torch.complex64) @ self.W_enc.T
        # # out = out[..., 0 : self.n_fft // 2 + 1]

        # import matplotlib.pyplot as plt
        # plt.matshow(np.log(out[0, 0].abs().data.cpu().numpy()).T, origin='lower', aspect='auto', cmap='jet')
        # plt.savefig("_zz.pdf") 
        # from IPython import embed; embed(using=False); os._exit(0)
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
        x_flip = torch.flip(x[..., 1 : -1], dims=[-1]).conj()
        x = torch.cat([x, x_flip], dim=-1)
        x = x @ self.W_dec.T

        # Overlap-add
        out = fold(
            x=rearrange(x, 'b c t n -> (b c) n t'), 
            hop_length=self.hop_length, 
            window=self.window
        ).real  # (b*c, l)
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