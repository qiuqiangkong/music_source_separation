from __future__ import annotations

import math

import librosa
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor, LongTensor
from torch.nn.utils.rnn import pad_sequence


class BandSplit(nn.Module):
    def __init__(
        self, 
        sr: float, 
        n_fft: int, 
        n_bands: int, 
        in_channels: int, 
        out_channels: int
    ) -> None:
        r"""Band split STFT to mel scale STFT.

        f: stft bins
        k: mel bins
        w: band width
        """

        super().__init__()

        self.sr = sr
        self.n_fft = n_fft
        self.n_bands = n_bands
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Init mel banks
        melbanks, ola_window = self.init_melbanks()
        self.register_buffer(name='melbanks', tensor=Tensor(melbanks))  # (k, f)
        self.register_buffer(name='ola_window', tensor=Tensor(ola_window))  # (f,)

        self.pre_bands = nn.ModuleList()
        self.post_bands = nn.ModuleList()

        for i in range(self.n_bands):
            indices = torch.nonzero(self.melbanks[i].abs() > 1e-6, as_tuple=True)[0]
            self.register_buffer(name=f"indices_{i}", tensor=LongTensor(indices))

            n_in = self.in_channels * len(indices)
            self.pre_bands.append(nn.Linear(n_in, self.out_channels))
            self.post_bands.append(nn.Linear(self.out_channels, n_in))

        self.streams = None

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

        B = x.shape[0]
        D = self.out_channels
        T = x.shape[2]
        F = self.n_bands

        out = torch.zeros((B, D, T, F), device=x.device)

        if self.streams is None:
            self.streams = [torch.cuda.Stream() for _ in range(self.n_bands)]

        for f in range(self.n_bands):

            stream = self.streams[f]
            with torch.cuda.stream(stream):
                indices = getattr(self, f"indices_{f}")
                y = rearrange(x[:, :, :, indices], 'b c t f -> b t (c f)')
                y = self.pre_bands[f](y)
                y = rearrange(y, 'b t d -> b d t')
                out[:, :, :, f] = y
            
        return out

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

        B = x.shape[0]
        D = self.in_channels
        T = x.shape[2]
        F = self.n_fft // 2 + 1

        out = torch.zeros((B, D, T, F), device=x.device)

        for f in range(self.n_bands):
            indices = getattr(self, f"indices_{f}")
            y = x[:, :, :, f]
            y = rearrange(y, 'b d t -> b t d')
            y = self.post_bands[f](y)
            y = rearrange(y, 'b t (c f) -> b c t f', f=len(indices))
            out.scatter_add_(dim=-1, index=indices[None, None, None, :].repeat(B, D, T, 1), src=y)
        
        # Divide overlap add window
        out /= self.ola_window

        return out


class BandLinear(nn.Module):
    def __init__(self, n_bands: int, in_channels: int, out_channels: int) -> None:
        r"""Band split linear layer."""

        super().__init__()

        self.weight = nn.Parameter(torch.empty(n_bands, in_channels, out_channels))  # (k, i, o)
        self.bias = nn.Parameter(torch.zeros(n_bands, out_channels))  # (k, o)

        bound = 1 / math.sqrt(in_channels)
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        r"""

        b: batch_size
        t: frames_num
        k: n_bands
        i: in_channels
        o: out_channels

        Args:
            x: (b, t, k, i)

        Returns:
            x: (b, t, k, o)
        """

        # Sum along the input channels axis
        return torch.einsum('btki,kio->btko', x, self.weight) + self.bias  # (b, t, k, o)


if __name__ == '__main__':

    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    x = torch.randn(8, 4, 201, 1025)  # (b, c, t, f)

    bandsplit = BandSplit(
        sr=44100, 
        n_fft=2048, 
        n_bands=64,
        in_channels=4, 
        out_channels=64
    )

    y = bandsplit.transform(x)  # (b, d, t, k)
    x_hat = bandsplit.inverse_transform(y)  # (b, c, t, f)
    print(x - x_hat)