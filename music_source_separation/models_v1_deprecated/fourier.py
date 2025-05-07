import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from einops import rearrange


class Fourier(nn.Module):
    
    def __init__(self, 
        n_fft=2048, 
        hop_length=441, 
        return_complex=True, 
        normalized=True
    ):
        super(Fourier, self).__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.return_complex = return_complex
        self.normalized = normalized

    def stft(self, waveform):
        """
        Args:
            waveform: (batch_size, channels_num, samples_num)

        Returns:
            complex_sp: (batch_size, channels_num, frames_num, freq_bins)
        """

        B, C, T = waveform.shape

        x = rearrange(waveform, 'b c t -> (b c) t')

        x = torch.stft(
            input=x, 
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft).to(x.device),
            normalized=self.normalized,
            return_complex=self.return_complex
        )
        # shape: (batch_size * channels_num, freq_bins, frames_num)

        complex_sp = rearrange(x, '(b c) f t -> b c t f', b=B, c=C)
        # shape: (batch_size, channels_num, frames_num, freq_bins)

        return complex_sp

    def istft(self, complex_sp):
        """
        Args:
            complex_sp: (batch_size, channels_num, frames_num, freq_bins)

        Returns:
            waveform: (batch_size, channels_num, samples_num)
        """

        B, C, T, F = complex_sp.shape

        x = rearrange(complex_sp, 'b c t f -> (b c) f t')

        x = torch.istft(
            input=x, 
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft).to(x.device),
            normalized=self.normalized,
        )
        # shape: (batch_size * channels_num, samples_num)

        x = rearrange(x, '(b c) t -> b c t', b=B, c=C)
        # shape: (batch_size, channels_num, samples_num)
        
        return x