from __future__ import annotations

import math

import librosa
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
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

        for f in range(self.n_bands):
            idxes = torch.nonzero(self.melbanks[f].abs() > 1e-6, as_tuple=True)[0]  # shape: (w,)
            pre_w = nn.Parameter(torch.zeros((in_channels, len(idxes), out_channels)))  # (i, f, o)
            post_w = nn.Parameter(torch.zeros((out_channels, in_channels, len(idxes))))  # (o, i, f)
            pre_b = nn.Parameter(torch.zeros((out_channels,)))  # (k, o)
            post_b = nn.Parameter(torch.zeros((in_channels, len(idxes))))  # (i, f)

            bound = 1 / math.sqrt(in_channels * (len(idxes)))
            nn.init.uniform_(pre_w, -bound, bound)

            bound = 1 / math.sqrt(out_channels)
            nn.init.uniform_(post_w, -bound, bound)

            debug = False
            if debug:
                nn.init.uniform_(pre_b, -1, 1)
                nn.init.uniform_(post_b, -1, 1)

            setattr(self, f"pre_w_{f}", pre_w)
            setattr(self, f"post_w_{f}", post_w)
            setattr(self, f"pre_b_{f}", pre_b)
            setattr(self, f"post_b_{f}", post_b)

            self.register_buffer(name=f'idxes_{f}', tensor=idxes)
            self.register_buffer(name=f'melbanks_{f}', tensor=self.melbanks[f, idxes])

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

        # x = rearrange(x, 'b d t f -> b t (d f)')

        # pre_w: (c, i, o), pre_b: (d)

        B, C, T, F_ = x.shape
        out = torch.zeros((B, self.out_channels, T, self.n_bands), device=x.device)  # (b, d, t, f)

        for f in range(self.n_bands):
            idxes = getattr(self, f"idxes_{f}")
            melbanks = getattr(self, f"melbanks_{f}")
            _x = x[:, :, :, idxes] * melbanks  # (b, c, t, i)
            _w = getattr(self, f"pre_w_{f}")  # (c, i, o)
            _b = getattr(self, f"pre_b_{f}")  # (o,)
            _y = torch.einsum('bcti,cio->bto', _x, _w) + _b  # (b, t, o)
            out[:, :, :, f] = rearrange(_y, 'b t o -> b o t')

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
        
        B, D, T, F = x.shape
        out = torch.zeros((B, self.in_channels, T, self.n_fft // 2 + 1), device=x.device)

        for f in range(self.n_bands):

            idxes = getattr(self, f"idxes_{f}")
            melbanks = getattr(self, f"melbanks_{f}")
            
            _x = x[:, :, :, f]  # (b, o, t)
            _w = getattr(self, f"post_w_{f}")  # (o, c, i)
            _b = getattr(self, f"post_b_{f}")  # (c, i)
            _y = torch.einsum('bot,oci->btci', _x, _w) + _b  # (b, t, c, i)
            _y *= melbanks
            _y = rearrange(_y, 'b t c i -> b c t i')
            out.scatter_add_(dim=3, index=idxes[None, None, None, :].repeat(B, self.in_channels, T, 1), src=_y)

        # Divide overlap add window
        out /= self.ola_window

        return out


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
    print(y.abs().mean())
    print(x_hat.abs().mean()) 
    from IPython import embed; embed(using=False); os._exit(0)