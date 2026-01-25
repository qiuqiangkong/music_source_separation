from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import librosa
import numpy as np
from mss.io.audio import load
from mss.io.crops import RandomCrop
from torch.utils.data import Dataset
from typing_extensions import Literal


class MUSDB18HQ_multi3(Dataset):
    r"""MUSDB18HQ [1] is a dataset containing 100 training audio files and 50 
    testing audio files, each with vocals, bass, drums, and other stems. The 
    total duration is 9.8 hours. The audio is stereo and sampled at 48,000 Hz. 
    After decompression, the dataset size is 30 GB.

    [1] https://zenodo.org/records/3338373

    The dataset looks like:

        musdb18hq (30 GB)
        ├── train (100 files)
        │   ├── A Classic Education - NightOwl
        │   │   ├── bass.wav
        │   │   ├── drums.wav
        │   │   ├── mixture.wav
        │   │   ├── other.wav
        │   │   └── vocals.wav
        │   ... 
        │   └── ...
        └── test (50 files)
            ├── Al James - Schoolboy Facination
            │   ├── bass.wav
            │   ├── drums.wav
            │   ├── mixture.wav
            │   ├── other.wav
            │   └── vocals.wav
            ... 
            └── ...
    """

    URL = "https://zenodo.org/records/3338373"

    DURATION = 35359.56  # Dataset duration (s), 9.8 hours, including training, 
    # valdiation, and testing

    def __init__(
        self,
        root: str = "/datasets/musdb18hq", 
        split: Literal["train", "test"] = "train",
        sr: int = 44100,
        crop: callable = RandomCrop(clip_duration=2.),
        target_stems: list[str] = ["vocals"],
        time_align: Literal["strict", "group", "random"] = "group",
        stem_transform: None | callable | list[callable] = None,
        group_transform: None | callable | list[callable] = None,
        mixture_transform: None | callable | list[callable] = None,
        min_resample_ratio: float = 0.95,
        max_resample_ratio: float = 1.05
    ) -> None:
        r"""
        time_align: str. "strict" indicates all stems are aligned (from the 
            same song and have the same start time). "group" indictates 
            target stems / background stems are aligned. "random" indicates 
            all stems are from different songs with different start time.
        """

        self.stems = ["bass", "drums", "other", "vocals"]
        self.root = root
        self.split = split
        self.sr = sr
        self.crop = crop
        self.target_stems = target_stems
        self.bg_stems = list(set(self.stems) - set(self.target_stems))
        self.time_align = time_align
        self.stem_transform = stem_transform
        self.group_transform = group_transform
        self.mixture_transform = mixture_transform

        self.segment_samples = int(self.crop.clip_duration * self.sr)
        self.segment_duration = self.crop.clip_duration

        if not Path(self.root).exists():
            raise Exception(f"{self.root} does not exist. Please download the dataset from {MUSDB18HQ.URL}")

        self.audios_dir = Path(self.root, self.split)
        self.list_names = sorted(os.listdir(self.audios_dir))
        self.audios_num = len(self.list_names)
        
    def __getitem__(
        self, 
        index_dict: dict,
    ) -> dict:

        # Use different song indexes for different stems
        index_dict = self.update_index_dict(index_dict)
        # E.g., {"bass": 94, "drums": 94, "other": 35, "vocals": 35}

        audio_names = {}
        audio_paths = {}
        start_times = {}
        # clip_durations = {}

        for stem in self.stems:

            audio_names[stem] = []
            audio_paths[stem] = []
            start_times[stem] = []
            # clip_durations[stem] = []

            for idx in index_dict[stem]:
                name = self.list_names[idx]
                path = str(Path(self.audios_dir, name, f"{stem}.wav"))
                audio_names[stem].append(name)
                audio_paths[stem].append(path)

                audio_duration = librosa.get_duration(path=path)
                start_time, clip_duration = self.crop(audio_duration=audio_duration)
                start_times[stem].append(start_time)
                # clip_durations[stem].append(clip_duration)

        start_times = self.update_start_times(start_times)
        # E.g., {"bass": 44.86, "drums": 139.68, "other": 44.86, "vocals": 139.68}

        data = {
            "dataset_name": "MUSDB18HQ",
        }

        for stem in self.stems:

            data[f"{stem}_audio_name"] = audio_names[stem]
            data[f"{stem}_audio_path"] = audio_paths[stem]
            data[f"{stem}_start_time"] = start_times[stem]

            mix_num = len(audio_paths[stem])
            data[stem] = np.zeros((mix_num, 2, self.segment_samples), dtype=np.float32)

            # Load a clip
            for i in range(mix_num):
                ratio = random.uniform(self.min_resample_ratio, self.max_resample_ratio)
                target_sr = round(ratio * self.sr)
                audio = load(
                    path=audio_paths[stem][i], 
                    sr=target_sr, 
                    offset=start_times[stem][i], 
                    duration=self.segment_duration,
                )
                # shape: (channels, audio_samples)

                data[stem][i] = audio

        # import soundfile
        # soundfile.write(file="_zz.wav", data=data["vocals"].T, samplerate=self.sr)
        # soundfile.write(file="_zz2.wav", data=out.T, samplerate=self.sr)
        # from IPython import embed; embed(using=False); os._exit(0)

        # Sum sources to target and background
        # data["target"], data["background"] = self.sources_to_target_and_background(data)

        # # Transform target and background
        # if self.group_transform is not None:
        #     data["target"] = self.group_transform(data["target"])
        #     data["background"] = self.group_transform(data["background"])

        # # Sum target and background to mixture
        # data["mixture"] = data["target"] + data["background"]

        # # Transform mixture
        # if self.mixture_transform is not None:
        #     data["mixture"] = self.mixture_transform["mixture"]

        return data

    def __len__(self) -> int:
        return self.audios_num

    def update_index_dict(self, index_dict: Union[int, dict]) -> dict:
        r"""Get song indexes of different stems."""

        if self.time_align == "strict":
            return {stem: index_dict[self.target_stems[0]] for stem in self.stems}

        elif self.time_align == "group":
            return {stem: index_dict[self.target_stems[0]] for stem in self.target_stems} | \
                {stem: index_dict[self.bg_stems[0]] for stem in self.bg_stems}

        elif self.time_align == "random":
            return index_dict

        else:
            raise TypeError(index_dict)

    def update_start_times(self, start_times: dict) -> dict:
        r"""Update start times according to different time_align types."""

        if self.time_align == "strict":
            return {stem: start_times[self.stems[0]] for stem in self.stems}
            
        elif self.time_align == "group":
            return {stem: start_times[self.target_stems[0]] for stem in self.target_stems} | \
                {stem: start_times[self.bg_stems[0]] for stem in self.bg_stems}
            
        elif self.time_align == "random":
            return start_times

        else:
            raise ValueError(self.time_align)