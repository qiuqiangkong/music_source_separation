from __future__ import annotations

import numpy as np


def filterbank_frequencies(
    sr: int, 
    n_freqs: int, 
    max_spacing: float,
    freq_scale="mel_linear"
) -> list[tuple]:
    """Generate mel-to-linear frequency points.

    Returns:
        freqs: (n_freqs,)
    """
    
    n_intervals = n_freqs - 1
    assert n_intervals * max_spacing >= sr / 2, "n_freqs or max_spacing is too small!"

    hz_to_scale, scale_to_hz = get_scale_fns(freq_scale)

    # Search for a valid transition
    for n in range(n_freqs, 0, -1):
        freqs = scale_to_hz(np.linspace(0, hz_to_scale(sr / 2), n))
        idx = np.flatnonzero(np.diff(freqs) > max_spacing)[0]

        n_linear = n_intervals - idx
        linear_spacing = (sr / 2 - freqs[idx]) / n_linear

        if linear_spacing <= max_spacing:
            break

    freqs = np.concatenate([
        freqs[0 : idx],  # scale banks
        freqs[idx] + np.arange(n_linear + 1) * linear_spacing  # linear banks
    ])

    return freqs


def erb_linear_freqs(n_bands: int, sr: int, freq_step: float) -> np.ndarray:

    edges = erb_to_hz(np.linspace(0, hz_to_erb(sr / 2), n_bands))
    freqs = [0.]
    idxes = [0]

    for i in range(1, n_bands):
        if edges[i] - edges[i - 1] < freq_step:
            freqs.append(edges[i])
            idxes.append(i)

        else:
            n = idxes.count(idxes[-1])

            while n > 0 or freqs[-1] + freq_step < edges[i]:
                if freqs[-1] + freq_step >= sr / 2:
                    break

                freqs.append(freqs[-1] + freq_step)
                idxes.append(i)
                n -= 1

    freqs.append(sr / 2)
    idxes.append(idxes[-1])

    return np.array(freqs), np.array(idxes)


def get_scale_fns(freq_scale: str) -> tuple[callable, callable]:
    if freq_scale == "mel_linear":
        return hz_to_mel, mel_to_hz
    elif freq_scale == "erb_linear":
        return hz_to_erb, erb_to_hz
    else:
        raise ValueError(f"Unsupported frequency scale: {freq_scale}")
    

def hz_to_mel(f: float) -> float:
    return 2595 * np.log10(1 + f / 700)


def mel_to_hz(mel: float) -> float:
    return 700 * (10 ** (mel / 2595) - 1)


def hz_to_erb(f: float, a=21.4, b=0.00437) -> float:
    return a * np.log10(1 + b * f)


def erb_to_hz(erb: float, a=21.4, b=0.00437) -> float:
    return 1 / b * (10 ** (erb / a) - 1)