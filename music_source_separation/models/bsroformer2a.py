from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from einops import rearrange
import numpy as np
import librosa
from dataclasses import dataclass
import math
import time

from music_source_separation.models.fourier import Fourier
from music_source_separation.models.attention import Block
from music_source_separation.models.rope import build_rope
from music_source_separation.models.bandsplit import BandSplit


@dataclass
class BSRoformerConfig:
    # STFT params
    audio_channels: int = 2
    sr: float = 44100
    n_fft: int = 2048
    hop_length: int = 441

    # Mel params
    mel_bins: int = 256
    mel_channels: int = 64

    # Transformer params
    patch_size: tuple[int, int] = (4, 4)
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 384


class BSRoformer2a(Fourier):
    def __init__(self, config: BSRoformerConfig) -> None:

        super(BSRoformer2a, self).__init__(
            n_fft=config.n_fft, 
            hop_length=config.hop_length, 
            return_complex=True, 
            normalized=True
        )

        self.cmplx_num = 2
        self.in_channels = config.audio_channels * self.cmplx_num
        self.fps = config.sr // config.hop_length

        self.patch_size = config.patch_size
        self.ds_factor = self.patch_size[0]
        self.head_dim = config.n_embd // config.n_head

        # STFT mel scale
        '''
        self.stft_mel = STFTMel(
            sr=config.sr, 
            in_channels=self.in_channels, 
            n_fft=config.n_fft, 
            mel_bins=config.mel_bins,
            out_channels=config.mel_channels
        )
        '''
        self.bandsplit = BandSplit(
            sr=config.sr, 
            n_fft=config.n_fft, 
            bands_num=config.mel_bins,
            in_channels=self.in_channels, 
            out_channels=config.mel_channels
        )

        # Patch STFT
        self.patch = Patch(
            in_channels=config.mel_channels * np.prod(self.patch_size),
            patch_size=self.patch_size,
            n_embd=config.n_embd
        )

        # Transformer blocks
        self.t_blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.f_blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))

        # Build RoPE cache
        t_rope = build_rope(seq_len=config.n_fft, head_dim=self.head_dim)
        f_rope = build_rope(seq_len=self.fps * 20, head_dim=self.head_dim)
        self.register_buffer(name="t_rope", tensor=t_rope)  # shape: (t, head_dim/2, 2)
        self.register_buffer(name="f_rope", tensor=f_rope)  # shape: (t, head_dim/2, 2)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Separation model.

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

        debug = "a3"

        # --- 1. Encode ---
        # 1.1 Complex spectrum
        complex_sp = self.stft(audio)  # shape: (b, c, t, f)

        x = torch.view_as_real(complex_sp)  # shape: (b, c, t, f, 2)
        x = rearrange(x, 'b c t f k -> b (c k) t f')  # shape: (b, d, t, f)

        if debug == "a1":
            x = x[:, :, 0:51, 0:64].repeat(1, 96, 1, 1)
        elif debug == "a2":
            # 1.2 Pad stft
            x, pad_t = self.pad_tensor(x)  # x: (b, d, t, f)

            # 1.3 Convert STFT to mel scale
            # x = self.stft_mel.transform(x)  # shape: (b, d, t, f)
            x = x[:, :, :, 0:256].repeat(1, 16, 1, 1)
            
            # 1.4 Patchify
            x = self.patch.patchify(x)  # shape: (b, d, t, f)

        elif debug == "a3":
            # 1.2 Pad stft
            x, pad_t = self.pad_tensor(x)  # x: (b, d, t, f)

            # 1.3 Convert STFT to mel scale
            t1 = time.time()
            x = self.bandsplit.transform(x)  # shape: (b, d, t, f)
            # print("b1", time.time() - t1)
            # x = x[:, :, :, 0:256].repeat(1, 16, 1, 1)
            
            # 1.4 Patchify
            x = self.patch.patchify(x)  # shape: (b, d, t, f)


        B = x.shape[0]  # batch size

        # --- 2. Transformer along time and frequency axes ---
        t1 = time.time()
        for t_block, f_block in zip(self.t_blocks, self.f_blocks):

            x = rearrange(x, 'b d t f -> (b f) t d')
            x = t_block(x, self.t_rope, mask=None)  # shape: (b*f, t, d)

            x = rearrange(x, '(b f) t d -> (b t) f d', b=B)
            x = f_block(x, self.f_rope, mask=None)  # shape: (b*t, f, d)

            x = rearrange(x, '(b t) f d -> b d t f', b=B)  # shape: (b, d, t, f)

        # print("b2", time.time() - t1)

        if debug == "a1":
            # 3.3 Get complex mask
            x = rearrange(x, 'b (c k) t f -> b c t f k', k=self.cmplx_num).contiguous()
            x = x.to(torch.float)  # compatible with bf16
            mask = torch.view_as_complex(x)  # shape: (b, c, t, f)

            mask = mask[:, 0:2, :, :].repeat(1, 1, 4, 17)[:, :, 0:201, 0:1025]

        elif debug == "a2":

            # --- 1. Decode ---
            # 3.1 Unpatchify
            x = self.patch.unpatchify(x)  # shape: (b, d, t, f)

            # 3.2 Convert mel scale STFT to original STFT
            # x = self.stft_mel.inverse_transform(x)  # shape: (b, d, t, f)
            x = x[:, 0:4, :, :].repeat(1, 1, 1, 5)[:, :, :, 0:1025]

            # 3.3 Get complex mask
            x = rearrange(x, 'b (c k) t f -> b c t f k', k=self.cmplx_num).contiguous()
            # x = x.to(torch.float)  # compatible with bf16
            mask = torch.view_as_complex(x)  # shape: (b, c, t, f)

            # 3.4 Unpad mask to the original shape
            mask = self.unpad_tensor(mask, pad_t)  # shape: (b, c, t, f)

        elif debug == "a3":

            # --- 1. Decode ---
            # 3.1 Unpatchify
            x = self.patch.unpatchify(x)  # shape: (b, d, t, f)

            # 3.2 Convert mel scale STFT to original STFT
            x = self.bandsplit.inverse_transform(x)  # shape: (b, d, t, f)
            # x = x[:, 0:4, :, :].repeat(1, 1, 1, 5)[:, :, :, 0:1025]

            # 3.3 Get complex mask
            x = rearrange(x, 'b (c k) t f -> b c t f k', k=self.cmplx_num).contiguous()
            # x = x.to(torch.float)  # compatible with bf16
            mask = torch.view_as_complex(x)  # shape: (b, c, t, f)

            # 3.4 Unpad mask to the original shape
            mask = self.unpad_tensor(mask, pad_t)  # shape: (b, c, t, f)

        # 3.5 Calculate stft of separated audio
        sep_stft = mask * complex_sp  # shape: (b, c, t, f)

        # 3.6 ISTFT
        output = self.istft(sep_stft)  # shape: (b, c, l)

        return output

    def pad_tensor(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Pad a spectrum that can be evenly divided by downsample_ratio.

        Args:
            x: E.g., (b, c, t=201, f)
        
        Outpus:
            output: E.g., (b, c, t=204, f)
        """

        # Pad last frames, e.g., 201 -> 204
        T = x.shape[2]
        pad_t = -T % self.ds_factor
        x = F.pad(x, pad=(0, 0, 0, pad_t))

        return x, pad_t

    def unpad_tensor(self, x: torch.Tensor, pad_t: int) -> torch.Tensor:
        """Unpad a spectrum to the original shape.

        Args:
            x: E.g., (b, c, t=204, f)
        
        Outpus:
            x: E.g., (b, c, t=201, f)
        """

        # Unpad last frames, e.g., 204 -> 201
        x = x[:, :, 0 : -pad_t, :]

        return x


class BandLinear(nn.Module):
    def __init__(self, bands_num, in_channels, out_channels):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(bands_num, in_channels, out_channels))
        self.bias = nn.Parameter(torch.zeros(bands_num, out_channels))

        bound = 1 / math.sqrt(in_channels)
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, x):
        r"""

        b: batch_size
        t: frames_num
        k: bands_num
        i: in_channels
        o: out_channels

        Args:
            x: (b, t, k, i)

        Returns:
            x: (b, t, k, o)
        """

        # self.weight: (k, i, o)
        # self.bias: (k, o)

        x = torch.einsum('btki,kio->btko', x, self.weight) + self.bias  # (b, t, k, o)

        return x

'''
class STFTMel(nn.Module):

    def __init__(
        self, 
        sr: float, 
        n_fft: int, 
        bands_num: int, 
        in_channels: int, 
        hid_channels: int
    ):
        super().__init__()

        self.sr = sr
        self.n_fft = n_fft
        self.bands_num = bands_num
        self.in_channels = in_channels
        self.hid_channels = hid_channels

        # Init mel banks
        melbanks, ola_window = self.init_melbanks()
        self.register_buffer(name='melbanks', tensor=torch.Tensor(melbanks))
        self.register_buffer(name='ola_window', tensor=torch.Tensor(ola_window))

        nonzero_indexes = []

        for f in range(self.bands_num):    
            idxes = (self.melbanks[f] != 0).nonzero(as_tuple=True)[0]
            nonzero_indexes.append(idxes)

        self.max_band_width = max([len(idxes) for idxes in nonzero_indexes])
        pad = self.max_band_width - 1

        f_idxes = pad_sequence(
            sequences=nonzero_indexes, 
            batch_first=True, 
            padding_value=pad
        )  # (k, wmax)
        f_idxes = f_idxes.flatten()  # (k*wmax,)
        mask = (f_idxes != pad).to(self.melbanks.dtype)  # (k*wmax,)
        
        self.register_buffer(name="f_idxes", tensor=f_idxes)
        self.register_buffer(name="mask", tensor=mask)

        self.pre_bandnet = BandLinear(
            bands_num=self.bands_num, 
            in_channels=self.max_band_width * self.in_channels, 
            out_channels=self.hid_channels
        )

        self.post_bandnet = BandLinear(
            bands_num=self.bands_num, 
            in_channels=self.hid_channels, 
            out_channels=self.max_band_width * self.in_channels
        )



    def init_melbanks(self) -> None:

        melbanks = librosa.filters.mel(
            sr=self.sr, 
            n_fft=self.n_fft, 
            n_mels=self.bands_num - 2, 
            norm=None
        )

        F = self.n_fft // 2 + 1

        # The zeroth bank, e.g., [1., 0., 0., ..., 0.]
        melbank_0 = np.zeros(F)
        melbank_0[0] = 1.

        # The last bank, e.g., [0., ..., 0., 0.18, 0.87, 1.]
        melbank_last = np.zeros(F)
        idx = np.argmax(melbanks[-1])
        melbank_last[idx :] = 1. - melbanks[-1, idx :]

        melbanks = np.concatenate(
            [melbank_0[None, :], melbanks, melbank_last[None, :]], axis=0
        )  # shape: (n_mels, f)

        ola_window = np.sum(melbanks, axis=0)  # overlap add window
        assert np.allclose(a=ola_window, b=1.)
        self.register_buffer(name="ola_window", tensor=torch.Tensor(ola_window))

        return melbanks, ola_window

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        r"""Convert STFT to mel scale.

        b: batch_size
        c: channels_num
        d: latent_dim
        t: frames_num
        q: nonzero values of a bank

        Args:
            x: (b, d, t, f)

        Returns:
            x: (b, d_out, t, k)
        """
        
        device = x.device
        B, D, T, Fq = x.shape

        x = x[:, :, :, self.f_idxes]  # (b, d, t, k*w')
        x *= self.mask  # (b, d, t, k*w')
        x = rearrange(x, 'b d t (k w) -> b t k (w d)', k=self.bands_num)  # (b, t, k, w*d)
        x = self.pre_bandnet(x)  # (b, t, k, d_out)        
        x = rearrange(x, 'b t k d -> b d t k')  # (b, d_out, t, k)

        return x

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        r"""Convert mel scale STFT to STFT.

        b: batch_size
        c: channels_num
        d: latent_dim
        t: frames_num
        q: nonzero values of a bank

        Args:
            x: (b, d, t, k)

        Outputs:
            y: (b, c, t, f)
        """

        x = rearrange(x, 'b d t k -> b t k d')  # (b, t, k, d_in)
        x = self.post_bandnet(x)  # (b, t, k, w*d)
        x = rearrange(x, 'b t k (w d) -> b d t (k w)', d=self.in_channels)  # (b, d, t, k*w)

        B, D, T, _ = x.shape
        Fq = self.n_fft // 2 + 1
        y = torch.zeros(B, D, T, Fq).to(x.device)  # shape: (b, d, t, f)

        x *= self.mask

        y.scatter_add_(
            dim=3, 
            index=self.f_idxes[None, None, None, :].repeat(B, D, T, 1),  # (b, d, t, k*w)
            src=x  # (b, d, t, k*w)
        )  # y: (b, d, t, f)

        # Divide overlap add window
        y /= self.ola_window

        # from IPython import embed; embed(using=False); os._exit(0)

        return y
'''

'''
class STFTMel(nn.Module):

    def __init__(
        self, 
        sr: float, 
        n_fft: int, 
        bands_num: int, 
        in_channels: int, 
        hid_channels: int
    ):
        super().__init__()

        self.sr = sr
        self.n_fft = n_fft
        self.bands_num = bands_num
        self.in_channels = in_channels
        self.hid_channels = hid_channels

        # Init mel banks
        melbanks, ola_window = self.init_melbanks()
        self.register_buffer(name='melbanks', tensor=torch.Tensor(melbanks))
        self.register_buffer(name='ola_window', tensor=torch.Tensor(ola_window))

        self.nonzero_indexes = []
        
        for f in range(self.bands_num):    
            idxes = (self.melbanks[f] != 0).nonzero(as_tuple=True)[0]
            self.nonzero_indexes.append(idxes)

        self.max_band_width = max([len(idxes) for idxes in self.nonzero_indexes])

        self.pre_bandnet = BandLinear(
            bands_num=self.bands_num, 
            in_channels=self.max_band_width * self.in_channels, 
            out_channels=self.hid_channels
        )

        self.post_bandnet = BandLinear(
            bands_num=self.bands_num, 
            in_channels=self.hid_channels, 
            out_channels=self.max_band_width * self.in_channels
        )

    def init_melbanks(self) -> None:

        melbanks = librosa.filters.mel(
            sr=self.sr, 
            n_fft=self.n_fft, 
            n_mels=self.bands_num - 2, 
            norm=None
        )

        F = self.n_fft // 2 + 1

        # The zeroth bank, e.g., [1., 0., 0., ..., 0.]
        melbank_0 = np.zeros(F)
        melbank_0[0] = 1.

        # The last bank, e.g., [0., ..., 0., 0.18, 0.87, 1.]
        melbank_last = np.zeros(F)
        idx = np.argmax(melbanks[-1])
        melbank_last[idx :] = 1. - melbanks[-1, idx :]

        melbanks = np.concatenate(
            [melbank_0[None, :], melbanks, melbank_last[None, :]], axis=0
        )  # shape: (n_mels, f)

        ola_window = np.sum(melbanks, axis=0)  # overlap add window
        assert np.allclose(a=ola_window, b=1.)
        self.register_buffer(name="ola_window", tensor=torch.Tensor(ola_window))

        return melbanks, ola_window

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        r"""Convert STFT to mel scale.

        b: batch_size
        c: channels_num
        d: latent_dim
        t: frames_num
        q: nonzero values of a bank

        Args:
            x: (b, d, t, f)

        Returns:
            x: (b, d_out, t, k)
        """
        
        device = x.device
        B, D, T, Fq = x.shape

        buffer = torch.zeros(B, D, T, self.bands_num, self.max_band_width).to(device)
        # buffer: (b, d, t, k, w')

        t1 = time.time()
        for k in range(self.bands_num):

            t1 = time.time()
            idxes = self.nonzero_indexes[k]  # (w,)
            print("d1", time.time() - t1)

            t1 = time.time()
            mel_bank = self.melbanks[k, idxes]  # (w,)
            print("d2", time.time() - t1)

            t1 = time.time()
            stft_bank = x[:, :, :, idxes]  # (b, d, t, w)
            print("d3", time.time() - t1)

            t1 = time.time()
            v = stft_bank * mel_bank  # (b, d, t, w)
            print("d4", time.time() - t1)

            t1 = time.time()
            buffer[:, :, :, k, 0 : len(idxes)] = v  # buffer: (b, d, t, k, w')
            print("d5", time.time() - t1)

        print("c1", time.time() - t1)

        buffer = rearrange(buffer, 'b d t k w -> b t k (w d)')  # (b, t, k, w*d)

        # t1 = time.time()
        x = self.pre_bandnet(buffer)  # (b, t, k, d_out)
        # print("c2", time.time() - t1)

        x = rearrange(x, 'b t k d -> b d t k')

        return x

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        r"""Convert mel scale STFT to STFT.

        b: batch_size
        c: channels_num
        d: latent_dim
        t: frames_num
        q: nonzero values of a bank

        Args:
            x: (b, d, t, k)

        Outputs:
            y: (b, c, t, f)
        """

        x = rearrange(x, 'b d t k -> b t k d')  # (b, t, k, d_in)
        x = self.post_bandnet(x)  # (b, t, k, d_out)

        x = rearrange(x, 'b t k (w d) -> b d t k w', d=self.in_channels)  # (b, d, t, k, w)

        B, D, T, K, W = x.shape
        Fq = self.n_fft // 2 + 1

        y = torch.zeros(B, D, T, Fq).to(x.device)
        # shape: (b, d, t, f)

        for k in range(self.bands_num):

            idxes = self.nonzero_indexes[k]
            v = x[:, :, :, k, 0 : len(idxes)]  # (b, d, t, w)
            y[..., idxes] += v  # (b, c, t, f)

        # Divide overlap add window
        y /= self.ola_window

        return y
'''

class Patch(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, n_embd: int):
        super().__init__()

        self.patch_size = patch_size

        self.fc_in = nn.Linear(in_features=in_channels, out_features=n_embd)
        self.fc_out = nn.Linear(in_features=n_embd, out_features=in_channels)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        r"""Patchify STFT. 
        E.g., (b, d_in, 256, 204) -> (b, d_emb, 64, 51)

        Args:
            x: (b, d_in, t, f)

        Outputs:
            x: (b, d_emb, t/t2, f/f2)
        """

        t2, f2 = self.patch_size

        x = rearrange(x, 'b d (t1 t2) (f1 f2) -> b t1 f1 (t2 f2 d)', t2=t2, f2=f2)
        x = self.fc_in(x)  # (b, t, f, d)
        x = rearrange(x, 'b t f d -> b d t f')

        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        r"""Unpatchify STFT.
        E.g., (b, d_emb, 64, 51) -> (b, d_in, 256, 204)

        Args:
            x: (b, d_emb, t/t2, f/t2)

        Outputs:
            x: (b, d_in, t, f)
        """
        
        t2, f2 = self.patch_size
        
        x = rearrange(x, 'b d t f -> b t f d')
        x = self.fc_out(x)  # (b, t, f, d)
        x = rearrange(x, 'b t1 f1 (t2 f2 d) -> b d (t1 t2) (f1 f2)', t2=t2, f2=f2)

        return x