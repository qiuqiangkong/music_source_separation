from __future__ import annotations

import multiprocessing as mp
mp.set_start_method("spawn", force=True)

import argparse
import os
from copy import deepcopy
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import wandb
# import trackio as wandb
from mss.utils import (parse_yaml, requires_grad, update_ema, LinearWarmUp, 
    separate_overlap_add, calculate_sdr)


def train(args) -> None:
    r"""Train a music source separation system."""

    # Arguments
    config_path = args.config
    wandb_log = not args.no_log
    filename = Path(__file__).stem
    
    # Configs
    configs = parse_yaml(config_path)
    device = configs["train"]["device"]
    valid_num = configs["validate"]["audios_num"]

    # Checkpoints directory
    config_name = Path(config_path).stem
    ckpts_dir = Path("./checkpoints", filename, config_name)
    Path(ckpts_dir).mkdir(parents=True, exist_ok=True)

    # Datasets
    train_dataset = get_dataset(configs, split="train")

    # Sampler
    train_sampler = get_sampler(configs, train_dataset)

    # Dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset, 
        batch_size=configs["train"]["batch_size_per_device"], 
        sampler=train_sampler,
        num_workers=configs["train"]["num_workers"], 
        pin_memory=True
    )

    # Model
    model = get_model(
        configs=configs, 
        ckpt_path=configs["train"]["resume_ckpt_path"]
    ).to(device)

    # EMA
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    ema.eval()  # EMA model should always be in eval mode

    # Loss function
    loss_fn = get_loss_fn(configs)

    # Optimizer
    optimizer, scheduler = get_optimizer_and_scheduler(
        configs=configs, 
        params=model.parameters()
    )

    # Logger
    if wandb_log:
        wandb.init(project="mss", name=f"{config_name}")

    # aa = AA(stems=train_dataset.stems)

    # Train
    for step, data in enumerate(tqdm(train_dataloader)):

        # ------ 1. Training ------
        # 1.1 Data
        target = data["target"].to(device)
        mixture = data["mixture"].to(device)

        # 1.1 Forward
        model.train()
        output = model(mixture)

        # 1.2 Loss
        loss = loss_fn(output=output, target=target)
        
        # 1.3 Optimize
        optimizer.zero_grad()  # Reset all parameter.grad to 0
        loss.backward()  # Update all parameter.grad
        optimizer.step()  # Update all parameters based on all parameter.grad
        scheduler.step()
        update_ema(ema, model, decay=0.999)

        if step % 100 == 0:
            print(loss)
        
        # ------ 2. Evaluation ------
        # 2.1 Evaluate
        if step % configs["train"]["test_every_n_steps"] == 0:

            train_sdr = validate(
                configs=configs,
                model=ema,
                split="train",
                audios_num=valid_num,
            )

            test_sdr = validate(
                configs=configs,
                model=ema,
                split="test",
                audios_num=valid_num,
            )

            if wandb_log:
                wandb.log(
                    data={
                        "train_sdr": train_sdr, 
                        "test_sdr": test_sdr,
                    },
                    step=step
                )

            print("====== Overall metrics ====== ")
            print(f"Train SDR: {train_sdr:.2f} dB")
            print(f"Test SDR: {test_sdr:.2f} dB")
        
        # 2.2 Save model
        if step % configs["train"]["save_every_n_steps"] == 0:
            
            ckpt_path = Path(ckpts_dir, f"step={step}_ema.pth")
            torch.save(ema.state_dict(), ckpt_path)
            print("Save model to {}".format(ckpt_path))

        if step == configs["train"]["training_steps"]:
            break


# from mss.augmentations.torch.gain import RandomGain
# from mss.augmentations.torch.pitch import RandomPitch
# from einops import rearrange

class AA:
    def __init__(self, stems):
        self.stems = stems
        self.random_gain = RandomGain(min_db=-6, max_db=6)
        self.random_pitch = RandomPitch(sr=48000, min_semitone=-1., max_semitone=1.)

    def __call__(self, data):
        for stem in self.stems:
            x = rearrange(data[stem], 'b m c l -> (b m) c l')
            x = self.random_gain(x)
            x = self.random_pitch(x)
            from IPython import embed; embed(using=False); os._exit(0)
        # self.random_gain(data[])


def get_dataset(
    configs: dict, 
    split: str
) -> Dataset:
    r"""Get datasets."""

    from mss.io.crops import RandomCrop

    assert split == "train"

    sr = configs["sample_rate"]
    segment_duration = configs["segment_duration"]
    target_stem = configs["target_stem"]
    ds = f"{split}_datasets"

    for name in configs[ds].keys():
    
        if name == "MUSDB18HQ":
            from mss.datasets.musdb18hq import MUSDB18HQ
            return MUSDB18HQ(
                root=configs[ds][name]["root"],
                split=configs[ds][name]["split"],
                sr=sr,
                crop=RandomCrop(clip_duration=segment_duration, end_pad=0.),
                target_stems=[target_stem],
                time_align=configs[ds][name]["time_align"],
                mixture_transform=None,
                group_transform=None,
                stem_transform=None
            )

        elif name == "MUSDB18HQIntraMix":

            from mss.datasets.musdb18hq_mix import MUSDB18HQIntraMix

            return MUSDB18HQIntraMix(
                root=configs[ds][name]["root"],
                split=configs[ds][name]["split"],
                sr=sr,
                crop=RandomCrop(clip_duration=configs["load_duration"], end_pad=0.),
                segment_duration=configs["segment_duration"],
                target_stems=[target_stem],
                min_intra_sources=configs["augmentation"]["cpu"]["mixing"]["intra_source"]["min_sources"],
                max_intra_sources=configs["augmentation"]["cpu"]["mixing"]["intra_source"]["max_sources"],
                time_align=configs["augmentation"]["cpu"]["mixing"]["extra_source"]["time_align"],
                mixture_transform=None,
                group_transform=None,
                stem_transform=get_stem_transform(configs),

            )

        else:
            raise ValueError(name)
            

def get_sampler(configs: dict, dataset: Dataset) -> Iterable:
    r"""Get sampler."""

    name = configs["sampler"]

    if name == "RandomSongSampler":
        from mss.samplers.random_song_sampler import RandomSongSampler
        return RandomSongSampler(dataset)

    elif name == "RandomSongSampler_multi":
        from mss.samplers.random_song_sampler_multi import RandomSongSampler_multi
        return RandomSongSampler_multi(dataset)

    elif name == "RandomSongSamplerMix":
        from mss.samplers.random_song_sampler_mix import RandomSongSamplerMix
        return RandomSongSamplerMix(dataset, configs["augmentation"]["cpu"]["mixing"]["intra_source"]["max_sources"])

    else:
        raise ValueError(name)


def get_stem_transform(configs):

    stem_transform = []

    for name, value in configs["augmentation"]["cpu"].items():
        
        if name == "gain" and value["enabled"]:
            from mss.augmentations.numpy.gain import RandomGain
            stem_transform.append(RandomGain(
                min_db=value["min_db"], 
                max_db=value["max_db"]
            ))

        elif name == "eq" and value["enabled"]:
            from mss.augmentations.numpy.eq import RandomEQ
            stem_transform.append(RandomEQ(
                min_db=value["min_db"], 
                max_db=value["max_db"],
                n_bands=value["n_bands"]
            ))

        elif name == "resample" and value["enabled"]:
            from mss.augmentations.numpy.resample import RandomResample
            stem_transform.append(RandomResample(
                sr=configs["sample_rate"],
                min_ratio=value["min_ratio"],
                max_ratio=value["max_ratio"]
            ))

    return stem_transform


def get_model(
    configs: dict, 
    ckpt_path: str
) -> nn.Module:
    r"""Initialize model."""

    name = configs["model"]["name"]

    if name == "BSRoformer":
        from mss.models.bsroformer import BSRoformer
        model = BSRoformer(**configs["model"])

    elif name == "BSRoformerMagPhase":
        from mss.models2.bsroformer03a import BSRoformerMagPhase
        model = BSRoformerMagPhase(**configs["model"])

    elif name == "BSRoformerMagPhase2":
        from mss.models2.bsroformer04a import BSRoformerMagPhase2
        model = BSRoformerMagPhase2(**configs["model"])

    elif name == "BSRoformerMagPhase3":
        from mss.models2.bsroformer04a2 import BSRoformerMagPhase3
        model = BSRoformerMagPhase3(**configs["model"])

    elif name == "BSRoformerMagPhase4":
        from mss.models2.bsroformer04a3 import BSRoformerMagPhase4
        model = BSRoformerMagPhase4(**configs["model"])

    elif name == "BSRoformerMagPhase5":
        from mss.models2.bsroformer04a4 import BSRoformerMagPhase5
        model = BSRoformerMagPhase5(**configs["model"])

    elif name == "BSRoformerMagPhase5b":
        from mss.models2.bsroformer14a import BSRoformerMagPhase5b
        model = BSRoformerMagPhase5b(**configs["model"])

    elif name == "BSRoformerMagPhase04a5":
        from mss.models2.bsroformer04a5 import BSRoformerMagPhase04a5
        model = BSRoformerMagPhase04a5(**configs["model"])

    elif name == "BSRoformerMagPhase04a6":
        from mss.models2.bsroformer04a6 import BSRoformerMagPhase04a6
        model = BSRoformerMagPhase04a6(**configs["model"])

    elif name == "BSRoformerMagPhase04a7":
        from mss.models2.bsroformer04a7 import BSRoformerMagPhase04a7
        model = BSRoformerMagPhase04a7(**configs["model"])

    elif name == "BSRoformerMagPhase04a8":
        from mss.models2.bsroformer04a8 import BSRoformerMagPhase04a8
        model = BSRoformerMagPhase04a8(**configs["model"])

    elif name == "BSRoformerMagPhase04a9":
        from mss.models2.bsroformer04a9 import BSRoformerMagPhase04a9
        model = BSRoformerMagPhase04a9(**configs["model"])

    elif name == "BSRoformerMagPhase04a9":
        from mss.models2.bsroformer04a9 import BSRoformerMagPhase04a9
        model = BSRoformerMagPhase04a9(**configs["model"])

    elif name == "BSRoformerMagPhase2White":
        from mss.models2.bsroformer04b import BSRoformerMagPhase2White
        model = BSRoformerMagPhase2White(**configs["model"])

    elif name == "BSRoformerMagPhase3White":
        from mss.models2.bsroformer04b2 import BSRoformerMagPhase3White
        model = BSRoformerMagPhase3White(**configs["model"])

    elif name == "BSRoformerTmp":
        from mss.models2.bsroformer_tmp import BSRoformerTmp
        model = BSRoformerTmp(**configs["model"])

    elif name == "BSRoformerMulSTFT":
        from mss.models2.bsroformer_07a import BSRoformerMulSTFT
        model = BSRoformerMulSTFT(**configs["model"])

    elif name == "BSRoformer11a":
        from mss.models2.bsroformer11a import BSRoformer11a
        model = BSRoformer11a(**configs["model"])

    elif name == "BSRoformer12a":
        from mss.models2.bsroformer12a import BSRoformer12a
        model = BSRoformer12a(**configs["model"])

    elif name == "BSRoformer15a":
        from mss.models2.bsroformer15a import BSRoformer15a
        model = BSRoformer15a(**configs["model"])

    elif name == "BSRoformer15b":
        from mss.models2.bsroformer15b import BSRoformer15b
        model = BSRoformer15b(**configs["model"])

    elif name == "BSRoformer16a":
        from mss.models2.bsroformer16a import BSRoformer16a
        model = BSRoformer16a(**configs["model"])

    elif name == "BSRoformer17a":
        from mss.models2.bsroformer17a import BSRoformer17a
        model = BSRoformer17a(**configs["model"])

    elif name == "BSRoformer17b":
        from mss.models2.bsroformer17b import BSRoformer17b
        model = BSRoformer17b(**configs["model"])

    elif name == "BSRoformer18a":
        from mss.models2.bsroformer18a import BSRoformer18a
        model = BSRoformer18a(**configs["model"])

    elif name == "BSRoformer19a":
        from mss.models2.bsroformer19a import BSRoformer19a
        model = BSRoformer19a(**configs["model"])

    elif name == "BSRoformer20a":
        from mss.models2.bsroformer20a import BSRoformer20a
        model = BSRoformer20a(**configs["model"])

    elif name == "BSRoformer20b":
        from mss.models2.bsroformer20b import BSRoformer20b
        model = BSRoformer20b(**configs["model"])

    elif name == "BSRoformer21a":
        from mss.models2.bsroformer21a import BSRoformer21a
        model = BSRoformer21a(**configs["model"])

    elif name == "BSRoformer21b":
        from mss.models2.bsroformer21b import BSRoformer21b
        model = BSRoformer21b(**configs["model"])

    elif name == "BSRoformer22a":
        from mss.models2.bsroformer22a import BSRoformer22a
        model = BSRoformer22a(**configs["model"])

    elif name == "BSRoformer22b":
        from mss.models2.bsroformer22b import BSRoformer22b
        model = BSRoformer22b(**configs["model"])

    elif name == "BSRoformer23a":
        from mss.models2.bsroformer23a import BSRoformer23a
        model = BSRoformer23a(**configs["model"])

    elif name == "BSRoformer24a":
        from mss.models2.bsroformer24a import BSRoformer24a
        model = BSRoformer24a(**configs["model"])

    elif name == "BSRoformer25a":
        from mss.models2.bsroformer25a import BSRoformer25a
        model = BSRoformer25a(**configs["model"])

    elif name == "BSRoformer25b":
        from mss.models2.bsroformer25b import BSRoformer25b
        model = BSRoformer25b(**configs["model"])

    elif name == "BSRoformer26a":
        from mss.models2.bsroformer26a import BSRoformer26a
        model = BSRoformer26a(**configs["model"])

    elif name == "BSRoformer26d":
        from mss.models2.bsroformer26d import BSRoformer26d
        model = BSRoformer26d(**configs["model"])

    elif name == "BSRoformer26e":
        from mss.models2.bsroformer26e import BSRoformer26e
        model = BSRoformer26e(**configs["model"])

    elif name == "BSRoformer27a":
        from mss.models2.bsroformer27a import BSRoformer27a
        model = BSRoformer27a(**configs["model"])

    elif name == "BSRoformer27b":
        from mss.models2.bsroformer27b import BSRoformer27b
        model = BSRoformer27b(**configs["model"])

    elif name == "BSRoformer28a":
        from mss.models2.bsroformer28a import BSRoformer28a
        model = BSRoformer28a(**configs["model"])

    elif name == "BSRoformer29a":
        from mss.models2.bsroformer29a import BSRoformer29a
        model = BSRoformer29a(**configs["model"])

    elif name == "BSRoformer29a":
        from mss.models2.bsroformer29a import BSRoformer29a
        model = BSRoformer29a(**configs["model"])

    elif name == "BSRoformer30b":
        from mss.models2.bsroformer30b import BSRoformer30b
        model = BSRoformer30b(**configs["model"])

    elif name == "BSRoformer31a":
        from mss.models2.bsroformer31a import BSRoformer31a
        model = BSRoformer31a(**configs["model"])

    elif name == "BSRoformer32a":
        from mss.models2.bsroformer32a import BSRoformer32a
        model = BSRoformer32a(**configs["model"])

    elif name == "BSRoformer33a":
        from mss.models2.bsroformer33a import BSRoformer33a
        model = BSRoformer33a(**configs["model"])

    elif name == "BSRoformer34a":
        from mss.models2.bsroformer34a import BSRoformer34a
        model = BSRoformer34a(**configs["model"])

    elif name == "BSRoformer35a":
        from mss.models2.bsroformer35a import BSRoformer35a
        model = BSRoformer35a(**configs["model"])

    elif name == "BSRoformer36a":
        from mss.models2.bsroformer36a import BSRoformer36a
        model = BSRoformer36a(**configs["model"])

    elif name == "BSRoformer36b":
        from mss.models2.bsroformer36b import BSRoformer36b
        model = BSRoformer36b(**configs["model"])

    elif name == "BSRoformer37a":
        from mss.models2.bsroformer37a import BSRoformer37a
        model = BSRoformer37a(**configs["model"])

    elif name == "BSRoformer38a":
        from mss.models2.bsroformer38a import BSRoformer38a
        model = BSRoformer38a(**configs["model"])

    elif name == "BSRoformer38b":
        from mss.models2.bsroformer38b import BSRoformer38b
        model = BSRoformer38b(**configs["model"])

    elif name == "BSRoformer39a":
        from mss.models2.bsroformer39a import BSRoformer39a
        model = BSRoformer39a(**configs["model"])

    elif name == "BSRoformer39a2":
        from mss.models2.bsroformer39a2 import BSRoformer39a2
        model = BSRoformer39a2(**configs["model"])

    elif name == "BSRoformer39b":
        from mss.models2.bsroformer39b import BSRoformer39b
        model = BSRoformer39b(**configs["model"]) 

    elif name == "BSRoformer40a":
        from mss.models2.bsroformer40a import BSRoformer40a
        model = BSRoformer40a(**configs["model"])

    elif name == "BSRoformer41a":
        from mss.models2.bsroformer41a import BSRoformer41a
        model = BSRoformer41a(**configs["model"])

    elif name == "BSRoformer42a":
        from mss.models2.bsroformer42a import BSRoformer42a
        model = BSRoformer42a(**configs["model"])

    elif name == "BSRoformer43a":
        from mss.models2.bsroformer43a import BSRoformer43a
        model = BSRoformer43a(**configs["model"])

    elif name == "BSRoformerTmp":
        from mss.models2.bsroformer_tmp import BSRoformerTmp
        model = BSRoformerTmp(**configs["model"])

    elif name == "BSRoformer45a":
        from mss.models2.bsroformer45a import BSRoformer45a
        model = BSRoformer45a(**configs["model"])

    elif name == "BSRoformer45b":
        from mss.models2.bsroformer45b import BSRoformer45b
        model = BSRoformer45b(**configs["model"])

    elif name == "BSRoformer47a":
        from mss.models2.bsroformer47a import BSRoformer47a
        model = BSRoformer47a(**configs["model"])

    elif name == "BSRoformer48a":
        from mss.models2.bsroformer48a import BSRoformer48a
        model = BSRoformer48a(**configs["model"])

    elif name == "BSRoformer49a":
        from mss.models2.bsroformer49a import BSRoformer49a
        model = BSRoformer49a(**configs["model"])

    elif name == "BSRoformer49b":
        from mss.models2.bsroformer49b import BSRoformer49b
        model = BSRoformer49b(**configs["model"])

    elif name == "BSRoformer49c":
        from mss.models2.bsroformer49c import BSRoformer49c
        model = BSRoformer49c(**configs["model"])

    elif name == "BSRoformer50a":
        from mss.models2.bsroformer50a import BSRoformer50a
        model = BSRoformer50a(**configs["model"])

    elif name == "BSRoformer51a":
        from mss.models2.bsroformer51a import BSRoformer51a
        model = BSRoformer51a(**configs["model"])

    elif name == "BSRoformer52a":
        from mss.models2.bsroformer52a import BSRoformer52a
        model = BSRoformer52a(**configs["model"])

    elif name == "BSRoformer53a":
        from mss.models2.bsroformer53a import BSRoformer53a
        model = BSRoformer53a(**configs["model"])

    elif name == "BSRoformer53b":
        from mss.models2.bsroformer53b import BSRoformer53b
        model = BSRoformer53b(**configs["model"])

    elif name == "BSRoformer54a":
        from mss.models2.bsroformer54a import BSRoformer54a
        model = BSRoformer54a(**configs["model"])

    elif name == "BSRoformer54b":
        from mss.models2.bsroformer54b import BSRoformer54b
        model = BSRoformer54b(**configs["model"])

    elif name == "BSRoformer55a":
        from mss.models2.bsroformer55a import BSRoformer55a
        model = BSRoformer55a(**configs["model"])

    elif name == "BSRoformer56a":
        from mss.models2.bsroformer56a import BSRoformer56a
        model = BSRoformer56a(**configs["model"])

    elif name == "BSRoformer57a":
        from mss.models2.bsroformer57a import BSRoformer57a
        model = BSRoformer57a(**configs["model"])

    elif name == "BSRoformer60b":
        from mss.models2.bsroformer60b import BSRoformer60b
        model = BSRoformer60b(**configs["model"])

    elif name == "BSRoformer61a":
        from mss.models2.bsroformer61a import BSRoformer61a
        model = BSRoformer61a(**configs["model"])

    else:
        raise ValueError(name)    

    if ckpt_path:
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt)

    return model


def get_loss_fn(configs: dict) -> callable:
    r"""Get loss function."""

    loss_type = configs["train"]["loss"]

    if loss_type == "l1":
        from mss.losses.l1 import l1
        return l1

    elif loss_type == "l1_wav_l1_multistft":
        from mss.losses.wav_stft import MultiResolutionSTFTLoss
        return MultiResolutionSTFTLoss()

    else:
        raise ValueError(loss_type)


def get_optimizer_and_scheduler(
    configs: dict, 
    params: list[torch.Tensor]
) -> tuple[optim.Optimizer, None | optim.lr_scheduler.LambdaLR]:
    r"""Get optimizer and scheduler."""

    lr = float(configs["train"]["lr"])
    warm_up_steps = configs["train"]["warm_up_steps"]
    optimizer_name = configs["train"]["optimizer"]

    if optimizer_name == "AdamW":
        optimizer = optim.AdamW(params=params, lr=lr)

    if warm_up_steps:
        lr_lambda = LinearWarmUp(warm_up_steps)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)
    else:
        scheduler = None

    return optimizer, scheduler
        

def validate(
    configs: dict,
    model: nn.Module,
    split: str,
    audios_num: None | int = None,
) -> float:
    r"""Validate the model on part of data.

    c: channels_num
    L: audio_samples
    """

    # root = configs[f"{split}_datasets"]["MUSDB18HQ"]["root"]
    root = f"./datasets/musdb18hq"
    sr = configs["sample_rate"]
    segment_duration = configs["segment_duration"]
    target_stem = configs["target_stem"]
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

    for idx in range(0, len(audio_names), skip_n):

        # Get data
        audio_name = audio_names[idx]    
        data = {}

        for stem in stems:
            audio_path = Path(audios_dir, audio_name, f"{stem}.wav")
            audio, _ = librosa.load(audio_path, sr=sr, mono=False)  # (c, L)
            data[stem] = audio

        data["mixture"] = np.sum([data[stem] for stem in stems], axis=0)  # (c, L)

        # Foward
        output = separate_overlap_add(
            model=model, 
            audio=data["mixture"], 
            segment_samples=segment_samples,
            hop_length=segment_samples,
            batch_size=batch_size
        )  # (c, L)
        
        sdr, _ = calculate_sdr(
            output=output, 
            target=data[target_stem], 
            sr=sr, 
        )
        
        print("{}/{}, {}, SDR: {:.2f} dB".format(idx, len(audio_names), audio_name, sdr))

        sdrs.append(sdr)

    return np.nanmedian(sdrs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path of config yaml.")
    parser.add_argument("--no_log", action="store_true", default=False)
    args = parser.parse_args()

    train(args)