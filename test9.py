import librosa
from librosa.core.convert import fft_frequencies, mel_frequencies
import numpy as np
import torch
import math
import time
import librosa
import matplotlib.pyplot as plt
import torchaudio
import soundfile


def add():

    sr = 44100
    n_fft = 2048
    n_mels = 256
    fmin = 0
    fmax = sr / 2
    htk = False

    melbanks = librosa.filters.mel(
        sr=sr, 
        n_fft=n_fft, 
        n_mels=n_mels, 
        norm=None
    )

    # weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=np.float32)

    # fftfreqs = fft_frequencies(sr=sr, n_fft=n_fft)
    # mel_f = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=htk)

    # fdiff = np.diff(mel_f)
    # ramps = np.subtract.outer(mel_f, fftfreqs)

    # for i in range(n_mels):
    #     # lower and upper slopes for all bins
    #     lower = -ramps[i] / fdiff[i]
    #     upper = ramps[i + 2] / fdiff[i + 1]

    #     # .. then intersect them with each other and zero
    #     weights[i] = np.maximum(0, np.minimum(lower, upper))

    F = n_fft // 2 + 1

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

    from IPython import embed; embed(using=False); os._exit(0)


def add2():

    a1 = torch.Tensor(np.arange(36).reshape((6,6)))
    a1[:, [1,2,3]]
    a1[[0,1,2,3],[0,1,2,3]]
    a1[[[0,1],[2,3]],[[0,1],[2,3]]]
    from IPython import embed; embed(using=False); os._exit(0)


def add3():

    from pathlib import Path
    from torch.utils.data import DataLoader

    from audidata.io.crops import RandomCrop
    from audidata.datasets import MUSDB18HQ
    from audidata.samplers import InfiniteSampler, MUSDB18HQ_RandomSongSampler

    root = "/datasets/musdb18hq"

    sr = 44100

    dataset = MUSDB18HQ(
        root=root,
        split="train",
        sr=sr,
        crop=RandomCrop(clip_duration=2., end_pad=0.),
        target_stems=["vocals", "drums"],
        time_align="group",
        mixture_transform=None,
        group_transform=None,
        stem_transform=None
    )

    print(dataset[3])
    print(dataset[{"vocals": 3, "bass": 11, "drums": 3, "other": 11}])

    sampler1 = InfiniteSampler(dataset)  # Mix stems from a same song.
    # sampler2 = MUSDB18HQ_RandomSongSampler(dataset)  # Mix stems from different songs. Better performance

    dataloader = DataLoader(dataset=dataset, batch_size=4, sampler=sampler1)

    for data in dataloader:
        print(data)
        target = data["target"][0].cpu().numpy()
        bg = data["background"][0].cpu().numpy()
        mixture = data["mixture"][0].cpu().numpy()

        target_stft = librosa.core.stft(y=target, n_fft=2048, hop_length=441, window='hann', center=True)
        mixture_stft = librosa.core.stft(y=mixture, n_fft=2048, hop_length=441, window='hann', center=True)

        mask = target_stft / mixture_stft
        mask = mask.flatten()

        import matplotlib.pyplot as plt
        tmp = mask[0::100]
        plt.scatter(np.real(tmp), np.imag(tmp), s=1)
        plt.savefig("_zz.pdf")


        fig, axs = plt.subplots(2,1, sharex=False)
        hist, bin_edges = np.histogram(np.abs(mask), bins=200, range=(0, 10))
        axs[0].stem(bin_edges[:-1], hist)
        hist, bin_edges = np.histogram(np.angle(mask), bins=200, range=(-math.pi, math.pi))
        axs[1].stem(bin_edges[:-1], hist)
        plt.savefig("_zz2.pdf")

        fig, axs = plt.subplots(2,1, sharex=False)
        hist, bin_edges = np.histogram(np.real(mask), bins=200, range=(-10, 10))
        axs[0].stem(bin_edges[:-1], hist)
        hist, bin_edges = np.histogram(np.imag(mask), bins=200, range=(-10, 10))
        axs[1].stem(bin_edges[:-1], hist)
        plt.savefig("_zz3.pdf")

        from IPython import embed; embed(using=False); os._exit(0)

        break

    import soundfile
    Path("results").mkdir(parents=True, exist_ok=True)
    soundfile.write(file="results/musdb18hq_target.wav", data=target.T, samplerate=sr)
    soundfile.write(file="results/musdb18hq_bg.wav", data=bg.T, samplerate=sr)
    soundfile.write(file="results/musdb18hq_mixture.wav", data=mixture.T, samplerate=sr)


def add4():

    from music_source_separation.models.rope import build_rope, apply_rope
    rope = build_rope(seq_len=1000, head_dim=32)

    fig, axs = plt.subplots(2, 1, sharex=True)
    axs[0].matshow(rope[:, :, 0].data.cpu().numpy().T, origin='lower', aspect='auto', cmap='jet')
    axs[1].matshow(rope[:, :, 1].data.cpu().numpy().T, origin='lower', aspect='auto', cmap='jet')
    plt.savefig("_zz.pdf")

    B = 4
    T = 100
    n_head = 16
    head_dim = 32
    x = torch.rand(B, T, n_head, head_dim)

    y = apply_rope(x, rope)

    from tmp_rope import build_rope as build_rope2
    from tmp_rope import apply_rope as apply_rope2

    rope2 = build_rope2(seq_len=1000, head_dim=32)
    y2 = apply_rope(x, rope2)

    from IPython import embed; embed(using=False); os._exit(0)





def add5():

    audio_path = "assets/vocals_accompaniment_10s.wav"
    audio, fs = librosa.load(path=audio_path, sr=44100, mono=True)

    audio = audio[0 : 44100 * 2]

    # while True:
    # t1 = time.time()
    # z = librosa.effects.pitch_shift(audio, sr=44100, n_steps=1.5, res_type="soxr_mq")
    # print(time.time() - t1)

    # soundfile.write(file="_zz.wav", data=, samplerate=44100)

    # t1 = time.time()
    # z = torchaudio.functional.pitch_shift(torch.Tensor(audio), sample_rate=44100, n_steps=1.5)
    # print(time.time() - t1)

    aug = RandomPitch(sr=44100)
    aug(audio)


def add6():

    x = 0
    for i in range(4, 64):
        x += i

    print(x)

    # bands_num = 

    f = 0
    interval = 4
    nonzero_indexes = []

    while f < 1025:
        end = min(f + interval, 1025)
        idxes = list(range(f, end))
        nonzero_indexes.append(idxes)
        f = end

        if interval < 32:
            interval += 1

    from IPython import embed; embed(using=False); os._exit(0)


def add7():

    buffer = torch.zeros(10)
    src = torch.Tensor([1, 2, 3, 4, 5, 6])

    buffer.scatter_add_(
        dim=0, 
        index=torch.LongTensor([3, 4, 5]), 
        src=src
    )

    print(buffer)

if __name__ == '__main__':

    add7()