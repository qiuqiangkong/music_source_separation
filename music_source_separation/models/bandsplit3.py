from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange
import math

import librosa
import numpy as np
from torch.nn.utils.rnn import pad_sequence


class BandSplit(nn.Module):
    def __init__(
        self, 
        sr: float, 
        n_fft: int, 
        bands_num: int, 
        in_channels: int, 
        out_channels: int
    ) -> None:
        r"""Band split STFT to mel scale STFT.

        f: stft bins
        k: mel bins
        w: band with
        """

        super().__init__()

        self.sr = sr
        self.n_fft = n_fft
        self.bands_num = bands_num
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Init mel banks
        melbanks, ola_window = self.init_melbanks()
        self.register_buffer(name='melbanks', tensor=Tensor(melbanks))  # (k, f)
        self.register_buffer(name='ola_window', tensor=Tensor(ola_window))  # (f,)

        nonzero_indexes = []  # shape: (k, w)
        nonzero_melbanks = []  # shape: (k, w)
        
        for f in range(self.bands_num):    
            # idxes = torch.nonzero(self.melbanks[f].abs() > 1e-6, as_tuple=True)[0]  # shape: (w,)
            idxes = self.melbanks[f].nonzero(as_tuple=True)[0]  # shape: (w,)
            nonzero_indexes.append(idxes)
            nonzero_melbanks.append(self.melbanks[f, idxes])
            print(f, len(idxes))

        max_band_width = max([len(idxes) for idxes in nonzero_indexes])
        pad = -1
        
        nonzero_indexes = pad_sequence(
            sequences=nonzero_indexes, 
            batch_first=True, 
            padding_value=pad
        )  # shape: (k, w)

        nonzero_melbanks = pad_sequence(
            sequences=nonzero_melbanks,
            batch_first=True,
            padding_value=0.
        )  # shape: (k, w)

        mask = torch.zeros_like(nonzero_melbanks)  # shape: (k, w)
        mask[torch.where(nonzero_melbanks != 0)] = 1
        
        new_pad = self.n_fft // 2
        nonzero_indexes[nonzero_indexes == pad] = new_pad
        
        self.register_buffer(name="nonzero_indexes", tensor=nonzero_indexes)
        self.register_buffer(name="nonzero_melbanks", tensor=nonzero_melbanks)
        self.register_buffer(name="mask", tensor=mask)

        self.pre_bandnet = BandLinear(
            bands_num=self.bands_num, 
            in_channels=max_band_width * self.in_channels, 
            out_channels=self.out_channels
        )

        self.post_bandnet = BandLinear(
            bands_num=self.bands_num, 
            in_channels=self.out_channels, 
            out_channels=max_band_width * self.in_channels
        )

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
            n_fft=self.n_fft, 
            n_mels=self.bands_num - 2, 
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
        assert np.allclose(a=ola_window, b=1., atol=0.1)

        return melbanks, ola_window

    def transform(self, x: Tensor) -> Tensor:
        r"""Convert STFT to mel scale STFT.

        b: batch_size
        c: channels_num
        d: latent_dim
        t: frames_num
        k: mel_bins
        w: max_bank_width

        Args:
            x: (b, c, t, f)

        Returns:
            x: (b, d, t, k)
        """

        # self.f_idxes: (k*w,)
        # self.mask: (k*w,)

        # Band split
        x = x[:, :, :, self.nonzero_indexes.flatten()]  # (b, c, t, k*w)
        x *= self.nonzero_melbanks.flatten() * self.mask.flatten()  # (b, c, t, k*w)
        x = rearrange(x, 'b c t (k w) -> b t k (w c)', k=self.bands_num)  # (b, t, k, w*c)

        # Apply individual MLP on each band
        x = self.pre_bandnet(x)  # (b, t, k, d)        
        x = rearrange(x, 'b t k d -> b d t k')  # (b, d, t, k)

        return x

    def inverse_transform(self, x: Tensor) -> Tensor:
        r"""Convert mel scale STFT to STFT.

        b: batch_size
        c: channels_num
        d: latent_dim
        t: frames_num
        k: mel_bins
        w: max_bank_width

        Args:
            x: (b, d, t, k)

        Outputs:
            y: (b, c, t, f)
        """
        
        # Apply individual MLP on each band
        x = rearrange(x, 'b d t k -> b t k d')  # (b, t, k, d_in)
        x = self.post_bandnet(x)  # (b, t, k, w*c)
        x = rearrange(x, 'b t k (w c) -> b c t (k w)', c=self.in_channels)  # (b, c, t, k*w)

        # Band combine
        B, C, T = x.shape[0 : 3]
        Fq = self.n_fft // 2 + 1
        buffer = torch.zeros(B, C, T, Fq).to(x.device)  # shape: (b, c, t, f)

        from IPython import embed; embed(using=False); os._exit(0)
        # Mask out
        x *= self.mask.flatten()
        
        # Scatter add
        buffer.scatter_add_(
            dim=3, 
            index=self.nonzero_indexes.flatten()[None, None, None, :].repeat(B, C, T, 1), 
            src=x
        )

        # Divide overlap add window
        buffer /= self.ola_window

        return buffer


class BandLinear(nn.Module):
    def __init__(self, bands_num: int, in_channels: int, out_channels: int) -> None:
        r"""Fast band split."""

        super().__init__()

        self.weight = nn.Parameter(torch.empty(bands_num, in_channels, out_channels))
        self.bias = nn.Parameter(torch.zeros(bands_num, out_channels))

        bound = 1 / math.sqrt(in_channels)
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        # Sum along the input channels axis
        x = torch.einsum('btki,kio->btko', x, self.weight) + self.bias  # (b, t, k, o)

        return x


if __name__ == '__main__':

    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    x = torch.randn(8, 4, 201, 1025)  # (b, c, t, f)

    bandsplit = BandSplit(
        sr=44100, 
        n_fft=2048, 
        bands_num=64,
        in_channels=4, 
        out_channels=64
    )

    y = bandsplit.transform(x)  # (b, d, t, k)
    x_hat = bandsplit.inverse_transform(y)  # (b, c, t, f)
    x - x_hat
    from IPython import embed; embed(using=False); os._exit(0)