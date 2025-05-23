from __future__ import annotations

import argparse
import os
from pathlib import Path

import math
import librosa
import numpy as np
import soundfile
import torch
import time

from music_source_separation.utils import calculate_sdr, parse_yaml
from train import get_model, separate
from scipy.signal import get_window

# np.set_printoptions(precision=2, suppress=True)


def evaluate(args) -> None:
    r"""Evaluate on the test set of MUSDB18HQ."""

    # Arguments and parameters
    config_yaml = args.config_yaml
    ckpt_path = args.ckpt_path
    results_dir = args.results_dir
    device = "cuda"
    split = "test"
    batch_size = 4

    # Default parameters
    configs = parse_yaml(config_yaml)
    sr = configs["sample_rate"]
    clip_duration = configs["clip_duration"]
    target_stem = configs["target_stem"]
    clip_samples = round(clip_duration * sr)

    # Model
    model = get_model(
        configs=configs, 
        ckpt_path=ckpt_path
    ).to(device)
    
    # Data paths
    root = configs[f"{split}_datasets"]["MUSDB18HQ"]["root"]
    audios_dir = Path(root, split)
    audio_names = sorted(os.listdir(audios_dir))

    # Data buffer
    stems = ["vocals", "bass", "drums", "other"]
    sdrs = []
    fast_sdrs = []

    for idx, audio_name in enumerate(audio_names):
    
        # Get data
        data = {}

        for stem in stems:
            audio_path = Path(audios_dir, audio_name, "{}.wav".format(stem))
            audio, _ = librosa.load(audio_path, sr=sr, mono=False)  # shape: (c, l)
            data[stem] = audio

        data["mixture"] = np.sum([data[stem] for stem in stems], axis=0)  # shape: (c, l)

        # Foward
        output = separate_overlap_add(
            model=model, 
            audio=data["mixture"], 
            clip_samples=clip_samples, 
            hop_length=round(0.25 * clip_samples),
            batch_size=batch_size
        )  # shape: (c, l)

        sdr = calculate_sdr(output=output, target=data[target_stem], sr=sr, mode="default")
        print("{}/{}, {}: {}".format(idx, len(audio_names), audio_name, sdr))
        sdrs.append(sdr)
        
        sdr = calculate_sdr(output=output, target=data[target_stem], sr=sr, mode="fast")
        print(idx, "Fast SDR: {}".format(sdr))
        fast_sdrs.append(sdr)

        if results_dir:
            
            # Write out separated audio
            output_path = Path(results_dir, audio_name, f"vocals.wav")
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            soundfile.write(file=output_path, data=output.T, samplerate=sr)
            print("Write out to {}".format(output_path))
        
    print("====== Overall metrics ====== ")
    print("SDR: {}".format(np.nanmedian(sdrs)))
    print("Fast SDR: {}".format(np.nanmedian(fast_sdrs)))


def calculate_sdr(
    output: np.ndarray, 
    target: np.ndarray, 
    sr: float, 
    mode: Literal["default", "fast"] = "default"
) -> float:
    r"""Calculate the SDR of separation result.

    Args:
        output: (c, l)
        target: (c, l)

    Returns:
        sdr: float
    """

    museval_sr = 44100
    output = librosa.resample(y=output, orig_sr=sr, target_sr=museval_sr)  # (c, l)
    target = librosa.resample(y=target, orig_sr=sr, target_sr=museval_sr)  # (c, l)

    if mode == "default":
        # Calculate SDR with official museval package
        import museval
        
        (sdrs, _, _, _) = museval.evaluate(
            references=target.T[None, :, :],  # shape: (sources_num, l, c)
            estimates=output.T[None, :, :]  # shape: (sources_num, l, c)
        )
        # from IPython import embed; embed(using=False); os._exit(0)
    elif mode == "fast":
        # Calculate SDR to speed up by 10 times.
        sdrs = fast_evaluate(
            references=target,  # shape: (c, l)
            estimates=output  # shape: (c, l)
        )
        # print(np.median(sdrs))
        # from IPython import embed; embed(using=False); os._exit(0)
    else:
        raise ValueError(mode)

    sdr = np.nanmedian(sdrs)

    return sdr


def fast_evaluate(
    references: np.ndarray, 
    estimates: np.ndarray, 
    win: int =1 * 44100, 
    hop: int =1 * 44100
):
    r"""Fast version to calculate SDR of separation result. This function is 
    200 times faster than museval.evaluate(). The error is within 0.001. 

    Args:
        output: (c, l)
        target: (c, l)

    Returns:
        sdr: float
    """

    refs = librosa.util.frame(references, frame_length=win, hop_length=hop)  # (c, t, n)
    ests = librosa.util.frame(estimates, frame_length=win, hop_length=hop)  # (c, t, n)

    segs_num = refs.shape[2]
    sdrs = []

    for n in range(segs_num):
        sdr = fast_sdr(ref=refs[:, :, n], est=ests[:, :, n])
        sdrs.append(sdr)

    return sdrs


def fast_sdr(
    ref: np.ndarray, 
    est: np.ndarray, 
    eps: float = 1e-10
):
    r"""Calcualte SDR.
    """
    noise = est - ref
    numerator = np.clip(a=np.mean(ref ** 2), a_min=eps, a_max=None)
    denominator = np.clip(a=np.mean(noise ** 2), a_min=eps, a_max=None)
    sdr = 10. * np.log10(numerator / denominator)
    return sdr


def separate_overlap_add(
    model: nn.Module, 
    audio: torch.Tensor, 
    clip_samples: int, 
    hop_length: int,
    batch_size: int
):
    r"""Split audio into clips. Separate each clip. Concatenate the results.

    b: batch_size
    c: channels_num
    t: cilp_samples
    n: clips_num

    Args:
        model: nn.Module
        audio: (c, audio_samples)
        clip_samples: int
        batch_size: int

    Returns:
        output: (c, audio_samples)
    """

    t1 = time.time()

    device = next(model.parameters()).device
    
    audio_samples = audio.shape[1]
    full_samples = clip_samples + math.ceil((audio_samples - clip_samples) / hop_length) * hop_length
    audio = librosa.util.fix_length(data=audio, size=full_samples, axis=-1)
    # shape: (c, n*t)

    window = get_window(window="hamming", Nx=clip_samples)
    
    clips = librosa.util.frame(
        audio, 
        frame_length=clip_samples, 
        hop_length=hop_length
    )  # shape: (c, t, n)

    clips = clips.transpose(2, 0, 1)  # shape: (n, c, t)
    clips = torch.Tensor(clips.copy()).to(device)
    clips_num = clips.shape[0]

    pointer = 0
    outputs = []

    while pointer < clips_num:

        batch_clips = torch.Tensor(clips[pointer : pointer + batch_size])
        # shape: (b, c, t)

        with torch.no_grad():
            model.eval()
            batch_output = model(batch_clips)
            batch_output = batch_output.cpu().numpy()  # shape: (b, c, t)

        outputs.append(batch_output)
        pointer += batch_size

    outputs = np.concatenate(outputs, axis=0)
    # shape: (n, c, t)

    y = np.zeros_like(audio)
    ola = np.zeros_like(audio)

    for i in range(clips_num):
        y[:, i * hop_length : i * hop_length + clip_samples] += outputs[i] * window
        ola[:, i * hop_length : i * hop_length + clip_samples] += window

    y = y / ola
    y = y[:, 0 : audio_samples]

    print("Sep time: {:.2f} s".format(time.time() - t1))

    return y


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_yaml', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--results_dir', type=str, required=True)

    args = parser.parse_args()

    evaluate(args)