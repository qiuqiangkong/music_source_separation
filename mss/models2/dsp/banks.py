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


def linear_banks(
    sr: int, 
    n_bands: int, 
) -> list[tuple[float, float]]:
    r"""Linear banks."""

    freqs = np.linspace(0, sr / 2, n_bands + 1)
    banks = [[freqs[i].item(), freqs[i + 1].item()] for i in range(len(freqs) - 1)]  # (k1+k2, 2)
    return banks


def mel_linear_banks2(
    sr: int, 
    n_bands: int, 
    max_bandwidth: float
) -> list[tuple[float, float]]:
    r"""Mel bank in low frequency and linear band in high frequency."""

    freqs = np.linspace(0, hz_to_mel(sr / 2), n_bands + 1)
    freqs = mel_to_hz(freqs)

    idx = np.argmax(np.diff(freqs) >= max_bandwidth)  # (k1,)
    mel_part = freqs[: idx + 1]  # (k1,)
    linear_part = np.arange(mel_part[-1] + max_bandwidth, sr//2 + max_bandwidth, max_bandwidth)  # (k2,)
    freqs = np.concatenate([mel_part, linear_part])  # (k1+k2,)
    freqs[-1] = sr // 2
    banks = [[freqs[i].item(), freqs[i + 1].item()] for i in range(len(freqs) - 1)]  # (k1+k2, 2)

    return banks


def erb_linear_banks(
    sr: int, 
    n_bands: int, 
    max_bandwidth: float
) -> list[tuple[float, float]]:
    r"""Mel bank in low frequency and linear band in high frequency."""

    freqs = np.linspace(0, hz_to_erb(sr / 2), n_bands + 1)
    freqs = erb_to_hz(freqs)

    idx = np.argmax(np.diff(freqs) >= max_bandwidth)  # (k1,)
    mel_part = freqs[: idx + 1]  # (k1,)
    linear_part = np.arange(mel_part[-1] + max_bandwidth, sr//2 + max_bandwidth, max_bandwidth)  # (k2,)
    freqs = np.concatenate([mel_part, linear_part])  # (k1+k2,)
    freqs[-1] = sr // 2
    banks = [[freqs[i].item(), freqs[i + 1].item()] for i in range(len(freqs) - 1)]  # (k1+k2, 2)

    return banks


def hz_to_erb(f):
    return 21.4 * np.log10(1 + 0.00437 * f)

def erb_to_hz(erb):
    return 1 / 0.00437 * (10 ** (erb / 21.4) - 1)


def hz_to_mel(f):
    return 2595 * np.log10(1 + f / 700)


def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)