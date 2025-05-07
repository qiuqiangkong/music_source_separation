import torch
import torch.nn as nn
from torch import Tensor

from music_source_separation.models.fourier import Fourier


def l1_loss(output: torch.Tensor, target: torch.Tensor) -> torch.float:
	r"""L1 loss."""
	return torch.mean(torch.abs(output - target))


class WavStft_L1(nn.Module):
	def __init__(self):
		super().__init__()

		self.fourier1 = Fourier(n_fft=2048, hop_length=441, return_complex=True, normalized=True)

	def __call__(self, output, target):

		stft_loss = (self.fourier1.stft(output) - self.fourier1.stft(target)).abs().mean()
		wav_loss = (output - target).abs().mean()

		total_loss = 1. * stft_loss + 1. * wav_loss
		
		return total_loss


class Wav_L1_Sdr(nn.Module):
	def __init__(self):
		super().__init__()

		pass

	def __call__(self, output, target):

		wav_loss = 1. * (output - target).abs().mean()

		sdr_loss = 0.
		win = 44100
		for i in range(2):
			loss = - fast_sdr(
				ref=target[:, :, i * win : (i + 1) * win], 
				est=output[:, :, i * win : (i + 1) * win]
			)
			sdr_loss += loss

		total_loss = 1. * wav_loss + 0.01 * sdr_loss
		
		return total_loss

		

def fast_sdr(
    ref: Tensor, 
    est: Tensor, 
    eps: float = 1e-10
):
    r"""Calcualte SDR.
    """
    noise = est - ref
    numerator = torch.clamp((ref ** 2).mean(), eps, None)
    denominator = torch.clamp((noise ** 2).mean(), eps, None)
    sdr = 10. * torch.log10(numerator / denominator)
    return sdr