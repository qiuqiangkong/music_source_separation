# Music Source Separation with Neural Networks

This repository contains a PyTorch implementation of music source separation systems using neural networks. The input to the model is a music mixture. The output can be separated vocals/bass/drums/other/background. The model architectures include UNet, BSRoformer, and others.

## 0. Install dependencies

```bash
# Clone the repo
git clone https://github.com/qiuqiangkong/music_source_separation
cd music_source_separation

# Install Python environment
conda create --name mss python=3.10

# Activate environment
conda activate mss

# Install Python packages dependencies
bash env.sh
```

## 1. Download dataset

```bash
bash ./scripts/download_musdb18hq.sh
```

The downloaded dataset after compression looks like:

<pre>
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
</pre>

## 2. Train

Takes \~3 hours on 1 RTX4090 to train for 100,000 steps.

```python
CUDA_VISIBLE_DEVICES=0 python train.py --config="./configs/small.yaml"
```

```python
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes 4 train_accelerate.py --config="./configs/small.yaml"
```

## 3. Inference

```python
CUDA_VISIBLE_DEVICES=0 python inference.py \
    --config="./configs/small.yaml" \
    --ckpt_path="./checkpoints/train/small/step=100000_ema.pth" \
    --audio_path="./assets/music_10s.wav" \
    --output_path="./out.wav"
```

## 4. Evaluate
```python
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
    --config="./configs/small.yaml" \
    --ckpt_path="./checkpoints/train/small/step=100000_ema.pth" 
```
