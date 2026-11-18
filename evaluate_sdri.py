from __future__ import annotations

import argparse
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile

from mss.utils import parse_yaml, separate_overlap_add, calculate_sdr
# from train2 import get_model
# from train3 import get_stem_transform
# from train import validate
import matplotlib.pyplot as plt
import random


def evaluate(args) -> None:
    r"""Evaluate on the test set of MUSDB18HQ."""

    # Arguments and parameters
    config_yaml = args.config_yaml
    ckpt_path = args.ckpt_path
    device = "cuda"

    # Default parameters
    configs = parse_yaml(config_yaml)

    # Model
    # model = get_model(
    #     configs=configs, 
    #     ckpt_path=ckpt_path
    # ).to(device)
    
    # Compute SDRs
    # sdr = validate(
    #     configs=configs,
    #     model=model,
    #     split="test",
    #     audios_num=None,
    #     hop_ratio=4
    # )

    # sdr = validate3(
    #     configs=configs,
    #     model=model,
    #     split="test",
    #     audios_num=None,
    #     hop_length=24000
    # )

    sdr = validate3(
        configs=configs,
        # model=model,
        split="test",
        audios_num=None,
        hop_length=24000
    )

    print("====== Overall metrics ====== ")
    print(f"Median SDR: {sdr:.2f} dB")


# gain and eq aug
def validate3(
    configs: dict,
    # model: nn.Module,
    split: str,
    audios_num: None | int = None,
    hop_length: int = None
) -> float:
    r"""Validate the model on part of data.

    c: channels_num
    L: audio_samples
    """

    root = configs[f"{split}_datasets"]["MUSDB18HQ"]["root"]
    sr = configs["sample_rate"]
    segment_duration = configs["segment_duration"]
    # target_stem = configs["target_stem"]
    target_stem = "other"
    batch_size = configs["train"]["batch_size_per_device"]
    segment_samples = round(segment_duration * sr)

    # Paths
    audios_dir = Path(root, split)
    audio_names = sorted(os.listdir(audios_dir))

    if audios_num:
        # Evaluate only part of data
        skip_n = max(1, len(audio_names) // audios_num)
    else:
        skip_n = 1
    
    stems = ["vocals", "bass", "drums", "other"]
    sdrs = []

    _configs = parse_yaml("./kqq_configs/70a.yaml")
    # transforms = get_stem_transform(_configs)

    for idx in range(0, len(audio_names), skip_n):

        # Get data
        audio_name = audio_names[idx]    
        data = {}

        for stem in stems:
            audio_path = Path(audios_dir, audio_name, f"{stem}.wav")
            audio, _ = librosa.load(audio_path, sr=sr, mono=False)  # (c, L)
            data[stem] = audio

        data["mixture"] = np.sum([data[stem] for stem in stems], axis=0)  # (c, L)

        # transforms = transforms[1 : 2]

        # x0 = data["mixture"].copy()
        # inv_datas = []
        # for transform in transforms:
        #     data["mixture"], inv_data = transform(data["mixture"])
        #     inv_datas.append(inv_data)

        # output = separate_overlap_add(
        #     model=model, 
        #     audio=data["mixture"], 
        #     segment_samples=segment_samples,
        #     hop_length=hop_length,
        #     batch_size=batch_size
        # )  # (c, L)

        # # output = data["mixture"]
        # for i in range(len(transforms) - 1, -1, -1):
        #     output = transforms[i].inverse(output, inv_datas[i])
        # # print(np.mean(np.abs(x0 - output)))
        
        sdr, _ = calculate_sdr(
            output=data["mixture"], 
            target=data[target_stem], 
            sr=sr, 
        )
        
        print("{}/{}, {}, SDR: {:.2f} dB".format(idx, len(audio_names), audio_name, sdr))

        sdrs.append(sdr)

    return np.nanmedian(sdrs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_yaml', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)

    args = parser.parse_args()

    evaluate(args)