import librosa
import numpy as np


def mel_linear_banks(
    sr: int, 
    n_bands: int, 
    max_bandwidth: float
) -> list[tuple[float, float]]:
    r"""Mel bank in low frequency and linear band in high frequency."""

    freqs = librosa.mel_frequencies(n_mels=n_bands + 1, fmin=0, fmax=sr//2)  # (k,)
    idx = np.argmax(np.diff(freqs) >= max_bandwidth)  # (k1,)
    mel_part = freqs[: idx + 1]  # (k1,)
    linear_part = np.arange(mel_part[-1] + max_bandwidth, sr//2 + max_bandwidth, max_bandwidth)  # (k2,)
    freqs = np.concatenate([mel_part, linear_part])  # (k1+k2,)
    freqs[-1] = sr // 2
    banks = [[freqs[i].item(), freqs[i + 1].item()] for i in range(len(freqs) - 1)]  # (k1+k2, 2)
    return banks