import argparse
import os
import random
from pathlib import Path
from typing import Optional, Union

import librosa
import museval
import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from copy import deepcopy

import wandb

wandb.require("core")

from data.audio import load
from data.musdb18hq import MUSDB18HQ
from data.crops import RandomCrop
from utils import update_ema, requires_grad


def train(args):

    # Arguments
    model_name = args.model_name
    clip_duration = args.clip_duration
    batch_size = args.batch_size
    lr = float(args.lr)

    # Default parameters
    sr = 44100
    mono = False
    num_workers = 16
    pin_memory = True
    use_scheduler = True
    test_step_frequency = 5000
    save_step_frequency = 10000
    evaluate_num = 10
    training_steps = 1000000
    wandb_log = True
    device = "cuda"

    filename = Path(__file__).stem
    source_types = MUSDB18HQ.source_types

    checkpoints_dir = Path("./checkpoints", filename, model_name)
    
    root = "/datasets/musdb18hq"

    # Training dataset
    train_dataset = MUSDB18HQ(
        root=root,
        split="train",
        sr=sr,
        crop=RandomCrop(clip_duration=clip_duration, end_pad=0.),
        remix={"no_remix": 0., "half_remix": 1.0, "full_remix": 0.}
    )

    # Samplers
    train_sampler = InfiniteSampler(train_dataset)

    # Dataloaders
    train_dataloader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        num_workers=num_workers, 
        pin_memory=pin_memory
    )

    # Model
    model = get_model(model_name)
    model.to(device)

    # EMA
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    ema.eval()  # EMA model should always be in eval mode

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    if use_scheduler:
        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=warmup_lambda)

    if wandb_log:
        config = vars(args) | {
            "filename": filename,
        }
        wandb.init(
            project="mini_source_separation2", 
            config=config, 
            name="{} {}".format(model_name, str(train_dataset.remix_weights)), 
            magic=True
        )

    # Create checkpoints directory
    Path(checkpoints_dir).mkdir(parents=True, exist_ok=True)

    # Train
    for step, data in enumerate(tqdm(train_dataloader)):

        mixture = data["mixture"].to(device)
        target = data["vocals"].to(device)

        # Forward
        model.train()
        output = model(mixture=mixture) 
        
        # Calculate loss
        loss = l1_loss(output, target)

        # Optimize
        optimizer.zero_grad()   # Reset all parameter.grad to 0
        loss.backward()     # Update all parameter.grad
        optimizer.step()    # Update all parameters based on all parameter.grad
        update_ema(ema, model)

        # Learning rate scheduler (optional)
        if use_scheduler:
            scheduler.step()
        
        # Evaluate
        if step % test_step_frequency == 0:

            sdrs = {}

            
            for split in ["train", "test"]:
            
                sdr = validate(
                    root=root, 
                    split=split, 
                    sr=sr,
                    clip_duration=clip_duration,
                    source_types=source_types, 
                    target_source_type="vocals",
                    batch_size=batch_size,
                    model=model,
                    evaluate_num=evaluate_num,
                )
                sdrs[split] = sdr
            
            sdr = validate(
                root=root, 
                split="test", 
                sr=sr,
                clip_duration=clip_duration,
                source_types=source_types, 
                target_source_type="vocals",
                batch_size=batch_size,
                model=ema,
                evaluate_num=evaluate_num,
            )
            sdrs["test_ema"] = sdr

            print("--- step: {} ---".format(step))
            print("Evaluate on {} songs.".format(evaluate_num))
            print("Loss: {:.3f}".format(loss))
            print("Train SDR: {:.3f}".format(sdrs["train"]))
            print("Test SDR: {:.3f}".format(sdrs["test"]))
            print("Test SDR ema: {:.3f}".format(sdrs["test_ema"]))

            if wandb_log:
                wandb.log(
                    data={
                        "train_sdr": sdrs["train"],
                        "test_sdr": sdrs["test"],
                        "test_sdr_ema": sdrs["test_ema"],
                        "loss": loss.item(),
                    },
                    step=step
                )
        
        # Save model.
        if step % save_step_frequency == 0:
            checkpoint_path = Path(checkpoints_dir, "step={}.pth".format(step))
            torch.save(model.state_dict(), checkpoint_path)
            print("Save model to {}".format(checkpoint_path))

            checkpoint_path = Path(checkpoints_dir, "latest.pth")
            torch.save(model.state_dict(), Path(checkpoint_path))
            print("Save model to {}".format(checkpoint_path))

            checkpoint_path = Path(checkpoints_dir, "latest_ema.pth")
            torch.save(ema.state_dict(), Path(checkpoint_path))
            print("Save model to {}".format(checkpoint_path))

        if step == training_steps:
            break


def get_model(model_name):
    if model_name == "UNet":
        from models.unet import UNet
        return UNet()
    elif model_name == "BSRoformer":
        from models.bs_roformer import BSRoformer
        return BSRoformer(
            time_stacks=1,
            depth=12,
            dim=384,
            n_heads=12,
        )
    elif model_name == "BSRoformer2":
        from models.bs_roformer2 import BSRoformer2
        return BSRoformer2(
            depth=12,
            dim=384,
            n_heads=12,
        )
    elif model_name == "BSRoformer3":
        from models.bs_roformer3 import BSRoformer3
        return BSRoformer3(
            depth=12,
            dim=384,
            n_heads=12,
        )
    elif model_name == "BSRoformer4a":
        from models.bs_roformer4 import BSRoformer4a
        return BSRoformer4a(
            depth=12,
            dim=384,
            n_heads=12,
        )
    elif model_name == "BSRoformer4b":
        from models.bs_roformer4 import BSRoformer4b
        return BSRoformer4b(
            depth=12,
            dim=384,
            n_heads=12,
        )
    elif model_name == "BSRoformer5a":
        from models.bs_roformer5 import BSRoformer5a
        return BSRoformer5a(
            depth=12,
            dim=384,
            n_heads=12,
        )
    elif model_name == "BSRoformer6a":
        from models.bs_roformer6 import BSRoformer6a
        return BSRoformer6a(
            depth=12,
            dim=384,
            n_heads=12,
        )
    elif model_name == "BSRoformer7a":
        from models.bs_roformer7 import BSRoformer7a
        return BSRoformer7a(
            n_fft=2048,
            hop_length=441,
            input_channels=2,
            t1=101,
            f1=64,
            t2=4,
            f2=1,
            depth=12,
            dim=384,
            n_heads=12
        )
    elif model_name == "BSRoformer7b":
        from models.bs_roformer7 import BSRoformer7a
        return BSRoformer7a(
            n_fft=2048,
            hop_length=441,
            input_channels=2,
            t1=201,
            f1=64,
            t2=1,
            f2=1,
            depth=12,
            dim=384,
            n_heads=12
        )
    elif model_name == "BSRoformer8a":
        from models.bs_roformer8 import BSRoformer8a
        return BSRoformer8a(
            n_fft=2048,
            hop_length=441,
            input_channels=2,
            t1=101,
            f1=64,
            t2=4,
            f2=1,
            depth=24,
            dim=384,
            n_heads=12
        )
    elif model_name == "BSRoformer9a":
        from models.bs_roformer9 import BSRoformer9a
        return BSRoformer9a(
            n_fft=2048,
            hop_length=441,
            input_channels=2,
            t1=101,
            f1=64,
            t2=4,
            f2=1,
            depth=24,
            dim=384,
            n_heads=12
        )
    elif model_name == "BSRoformer10a":
        from models.bs_roformer10 import BSRoformer10a
        return BSRoformer10a(
            n_fft=2048,
            hop_length=441,
            input_channels=2,
            depth=24,
            dim=384,
            n_heads=12
        )
    elif model_name == "BSRoformer10b":
        from models.bs_roformer10 import BSRoformer10a
        return BSRoformer10a(
            n_fft=2048,
            hop_length=441,
            input_channels=2,
            depth=12,
            dim=384,
            n_heads=12
        )
    elif model_name == "BSRoformer11a":
        from models.bs_roformer11 import BSRoformer11a
        return BSRoformer11a(
            n_fft=2048,
            hop_length=441,
            input_channels=2,
            depth=12,
            dim=384,
            n_heads=12
        )
    elif model_name == "BSRoformer12a":
        from models.bs_roformer12 import BSRoformer12a
        return BSRoformer12a(
            n_fft=2048,
            hop_length=441,
            input_channels=2,
            depth=12,
            dim=384,
            n_heads=12
        )
    elif model_name == "BSRoformer13a":
        from models.bs_roformer13 import BSRoformer13a
        return BSRoformer13a(
            input_channels=2,
            depth=12,
            dim=384,
            n_heads=12
        )
    elif model_name == "BSRoformer14a":
        from models.bs_roformer14 import BSRoformer14a
        return BSRoformer14a(
            input_channels=2,
            depth=24,
            dim=384,
            n_heads=12
        )
    elif model_name == "BSRoformer15a":
        from models.bs_roformer15 import BSRoformer15a
        return BSRoformer15a(
            input_channels=2,
            depth=12,
            dim=384,
            n_heads=12
        )
    elif model_name == "BSRoformer16a":
        from models.bs_roformer16 import BSRoformer16a
        return BSRoformer16a(
            input_channels=2,
            depth=12,
            dim=384,
            n_heads=12
        )
    elif model_name == "BSRoformer17a":
        from models.bs_roformer17 import BSRoformer17a
        return BSRoformer17a(
            input_channels=2,
        )
    elif model_name == "BSRoformer17b":
        from models.bs_roformer17 import BSRoformer17b
        return BSRoformer17b(
            input_channels=2,
        )
    elif model_name == "BSRoformer18a":
        from models.bs_roformer18 import BSRoformer18a
        return BSRoformer18a(
            input_channels=2,
        )
    elif model_name == "BSRoformer19a":
        from models.bs_roformer19 import BSRoformer19a
        return BSRoformer19a(
            input_channels=2,
        )
    elif model_name == "BSRoformer20a":
        from models.bs_roformer20 import BSRoformer20a
        return BSRoformer20a(
            input_channels=2,
        )
    elif model_name == "BSRoformer21a":
        from models.bs_roformer21 import BSRoformer21a
        return BSRoformer21a(
            input_channels=2,
        )
    elif model_name == "WavUNet":
        from models.wavunet import WavUNet
        return WavUNet()
    else:
        raise NotImplementedError


class InfiniteSampler:
    def __init__(self, dataset):

        self.indexes = list(range(len(dataset)))
        random.shuffle(self.indexes)
        
    def __iter__(self):

        pointer = 0

        while True:

            if pointer == len(self.indexes):
                random.shuffle(self.indexes)
                pointer = 0
                
            index = self.indexes[pointer]
            pointer += 1

            yield index


def warmup_lambda(step, warm_up_steps=1000):
    if step <= warm_up_steps:
        return step / warm_up_steps
    else:
        return 1.


def l1_loss(output, target):
    return torch.mean(torch.abs(output - target))



def validate(
    root: str, 
    split: Union["train", "test"], 
    sr: int, 
    clip_duration: float, 
    source_types: list, 
    target_source_type: str, 
    batch_size: int, 
    model: nn.Module, 
    evaluate_num: Optional[int],
    verbose: bool = False
) -> float:
    r"""Calculate SDR.
    """

    clip_samples = round(clip_duration * sr)

    audios_dir = Path(root, split)
    audio_names = sorted(os.listdir(audios_dir))

    all_sdrs = []

    if evaluate_num:
        audio_names = audio_names[0 : evaluate_num]

    for audio_name in tqdm(audio_names):

        data = {}

        for source_type in source_types:
            audio_path = Path(audios_dir, audio_name, "{}.wav".format(source_type))

            audio = load(
                audio_path,
                sr=sr,
                mono=False
            )
            # shape: (channels, audio_samples)

            data[source_type] = audio

        data["mixture"] = np.sum([
            data[source_type] for source_type in source_types], axis=0)

        sep_wav = separate(
            model=model, 
            audio=data["mixture"], 
            clip_samples=clip_samples,
            batch_size=batch_size
        )

        target_wav = data[target_source_type]

        # Calculate SDR. Shape should be (sources_num, channels_num, audio_samples)
        (sdrs, _, _, _) = museval.evaluate([target_wav.T], [sep_wav.T])

        sdr = np.nanmedian(sdrs)
        all_sdrs.append(sdr)

        if verbose:
            print(audio_name, "{:.2f} dB".format(sdr))

    sdr = np.nanmedian(all_sdrs)

    return sdr

        

def separate(
    model: nn.Module, 
    audio: np.ndarray, 
    clip_samples: int, 
    batch_size: int
) -> np.ndarray:
    r"""Separate a long audio.
    """

    device = next(model.parameters()).device

    audio_samples = audio.shape[1]
    padded_audio_samples = round(np.ceil(audio_samples / clip_samples) * clip_samples)
    audio = librosa.util.fix_length(data=audio, size=padded_audio_samples, axis=-1)

    clips = librosa.util.frame(
        audio, 
        frame_length=clip_samples, 
        hop_length=clip_samples
    )
    # shape: (channels_num, clip_samples, clips_num)
    
    clips = clips.transpose(2, 0, 1)
    # shape: (clips_num, channels_num, clip_samples)

    clips_num = clips.shape[0]

    pointer = 0
    outputs = []

    while pointer < clips_num:

        batch_clips = torch.Tensor(clips[pointer : pointer + batch_size].copy()).to(device)

        with torch.no_grad():
            model.eval()
            batch_output = model(mixture=batch_clips)
            batch_output = batch_output.cpu().numpy()

        outputs.append(batch_output)
        pointer += batch_size

    outputs = np.concatenate(outputs, axis=0)
    # shape: (clips_num, channels_num, clip_samples)

    channels_num = outputs.shape[1]
    outputs = outputs.transpose(1, 0, 2).reshape(channels_num, -1)
    # shape: (channels_num, clips_num * clip_samples)

    outputs = outputs[:, 0 : audio_samples]
    # shape: (channels_num, audio_samples)

    return outputs


# Not used.
def calculate_sdr(ref, est):
    eps = 1e-12
    s_true = ref
    s_artif = est - ref
    numerator = np.sum(s_true ** 2) + eps
    denominator = np.sum(s_artif ** 2) + eps
    sdr = 10. * (np.log10(numerator) - np.log10(denominator))

    return sdr


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="UNet")
    parser.add_argument('--clip_duration', type=float, default=2.0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', default=0.001)
    args = parser.parse_args()

    train(args)