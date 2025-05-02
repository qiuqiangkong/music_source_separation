
CUDA_VISIBLE_DEVICES=0 python train.py --config="./kqq_configs/01a.yaml" --no_log

CUDA_VISIBLE_DEVICES=0 python inference.py \
	--config="./kqq_configs/02a.yaml" \
	--ckpt_path="./checkpoints/train/02a/step=0.pth" \
	--audio_path="./assets/vocals_accompaniment_10s.wav" \
	--output_path="_zz.wav"


CUDA_VISIBLE_DEVICES=0 python evaluate.py \
	--config="./kqq_configs/02a.yaml" \
	--ckpt_path="./checkpoints/train/02a/step=0.pth" \
	--results_dir=""

CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --multi_gpu --num_processes 4 train_accelerate.py --config="./kqq_configs/01a.yaml" --no_log


# 04a.yaml    UNet, InfiniteSampler
# 04b.yaml    UNet, MUSDB18HQ_RandomSongSampler
# 04c.yaml    BSRoformer, MUSDB18HQ_RandomSongSampler
# 04d.yaml    BSRoformer, gpu=8, bf16, bs=8 MUSDB18HQ_RandomSongSampler
# 05a.yaml    UNet, InfiniteSampler
# 05b.yaml    UNet, MUSDB18HQ_RandomSongSampler, bs=16
# 06a.yaml	  BSRoformer, t=201, f=256, patch=(4,4), gpu=1, bs=4 MUSDB18HQ_RandomSongSampler
# 06b.yaml	  BSRoformer, t=201, f=256, patch=(4,4), gpu=4, bs=8 MUSDB18HQ_RandomSongSampler
# 07a.yaml	  BSRoformer, t=201, f=256, patch=(1,1), gpu=1, bs=4 MUSDB18HQ_RandomSongSampler
# 09a.yaml    BSRoformer, t=201, f=64, patch=(1,1), gpu=1, bs=4 MUSDB18HQ_RandomSongSampler
# 10a.yaml    BSRoformer, t=201, f=256, patch=(4,4), gpu=1, bs=4 MUSDB18HQ_RandomSongSampler

# 11a.yaml	  train2.py, Use oracle phase. others same as 10a, tr=13.5, te=12.0