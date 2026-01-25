from __future__ import annotations

import os
from pathlib import Path
from typing import Union
from torch import Tensor
import random

import librosa
import numpy as np
from mss.io.audio import load
from mss.io.crops import RandomCrop
from torch.utils.data import Dataset
from typing_extensions import Literal


class MUSDB18HQIntraMix(Dataset):
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
        crop: callable = RandomCrop(clip_duration=3.),
        segment_duration: float=2.0,
        target_stems: list[str] = ["vocals"],
        min_intra_sources: int = 1,
        max_intra_sources: int = 3,
        time_align: Literal["strict", "group", "random"] = "group",
        stem_transform: None | callable | list[callable] = None,
        group_transform: None | callable | list[callable] = None,
        mixture_transform: None | callable | list[callable] = None
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
        self.load_duration = crop.clip_duration
        self.segment_duration = segment_duration
        self.target_stems = target_stems
        self.bg_stems = list(set(self.stems) - set(self.target_stems))
        self.min_intra_sources = min_intra_sources
        self.max_intra_sources = max_intra_sources
        self.time_align = time_align
        self.stem_transform = stem_transform
        self.group_transform = group_transform
        self.mixture_transform = mixture_transform

        self.segment_samples = int(sr * segment_duration)
        self.ac = 2  # audio channels

        if not Path(self.root).exists():
            raise Exception(f"{self.root} does not exist. Please download the dataset from {MUSDB18HQ.URL}")

        self.audios_dir = Path(self.root, self.split)
        self.list_names = sorted(os.listdir(self.audios_dir))
        self.audios_num = len(self.list_names)
       
    def __getitem__(
        self, 
        index_dict: dict,
    ) -> dict:

        index_dict = self.update_index_dict(index_dict)
        
        audio_names = {stem: [] for stem in self.stems}
        audio_paths = {stem: [] for stem in self.stems}
        start_times = {stem: [] for stem in self.stems}
        
        # Prepare meta
        for stem in self.stems:

            for idx in index_dict[stem]:
                name = self.list_names[idx]
                path = str(Path(self.audios_dir, name, f"{stem}.wav"))
                audio_names[stem].append(name)
                audio_paths[stem].append(path)

                audio_duration = librosa.get_duration(path=path)
                start_time, _ = self.crop(audio_duration=audio_duration)
                start_times[stem].append(start_time)

        start_times = self.update_start_times(start_times)

        data = {
            "dataset_name": "MUSDB18HQ",
        }

        # Load and process audio
        for stem in self.stems:

            mix_num = len(audio_paths[stem])
            data[stem] = np.zeros((mix_num, self.ac, self.segment_samples), dtype=np.float32)  # (m, c, l)

            # Load a clip
            for i in range(mix_num):
                audio = load(
                    path=audio_paths[stem][i], 
                    sr=self.sr, 
                    offset=start_times[stem][i], 
                    duration=self.load_duration,
                )  # (channels, audio_samples)


                if self.stem_transform is not None:
                    for transform in self.stem_transform:
                        audio = transform(audio)

                data[stem][i] = audio[:, 0 : self.segment_samples]

            data[f"{stem}_audio_name"] = audio_names[stem]
            data[f"{stem}_audio_path"] = audio_paths[stem]
            data[f"{stem}_start_time"] = start_times[stem]

        # Sum sources to target and background
        data["target"], data["background"] = self.sources_to_target_and_background(data)

        # Sum target and background to mixture
        data["mixture"] = data["target"] + data["background"]

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

    def sources_to_target_and_background(self, data: dict) -> tuple[np.ndarray, np.ndarray]:
        r"""Sum sources to target and background."""

        target = 0
        bg = 0

        for stem in self.target_stems:
            n_sources = random.randint(self.min_intra_sources, self.max_intra_sources)
            target += np.sum(data[stem][0 : n_sources], axis=0)

        for stem in self.bg_stems:
            n_sources = random.randint(self.min_intra_sources, self.max_intra_sources)
            bg += np.sum(data[stem][0 : n_sources], axis=0)

        return target, bg