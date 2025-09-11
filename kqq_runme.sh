
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
# + 10a.yaml    BSRoformer, t=201, f=256, patch=(4,4), gpu=1, bs=4 MUSDB18HQ_RandomSongSampler
# + 10a_4gpus.yaml    gpu=4, bs=8, others same as 10a, 1-day 200k-steps.
# 10b_4gpus.yaml    gpu=4, bs=8, patch=(1, 4), te=8.86 after 1M steps 4 days (Must have problems!)
# + 11a.yaml	  train2.py, Use oracle phase. others same as 10a, gpu=1, tr=13.5, te=12.0 (3 day)
# 12a.yaml	  Use mixture phase. others same as 10a, gpu=1 (6 dB)
# 13a.yaml	  Oracle phase, wav_stft_l1 loss, others same as 11a (not better than l1)
# 14a.yaml	  Oracle phase, wav_l1_sdr, others same as 11a (not better than l1)

# 15a.yaml	  Cnn-patchify, others same as 10a
# 16a.yaml	  Gru-patchify, others same as 10a
# 16b.yaml	  Gru2-patchify, others same as 10a
# + 17a.yaml  individual mag and phase, others same as 10a, better than 10a
# 18a.yaml	  train2, uconnection, others same as 11a, no better than 11a
# 19a.yaml	  train2, uTransformer, others same as 11a, no better than 11a, maybe too small?
# 20a.yaml	  train2, transformer before patch, others same as 11a
# 21a.yaml	  train2, DC-AC (Junyan Chen), same as 11a.
# 21b.yaml    train2, DC-AC, compress 8x
# 21c.yaml    train2, DC-AC, compress 16x
# v1_deprecated_15a_4gpus.yaml    Previous, but still te=8.86. I am thinking the sampler has problem.
# v1_deprecated_15a_4gpus_02.yaml    Infinite sampler. Not its problem.

# 破案了！SDR eval "fast" 比 "default" 通常低 1.3 dB。

# + 22a_4gpus.yaml  the same as 10a_4gpus, BSRoformer, t=201, f=256, patch=(4,4), gpu=4, bs=8, fast_tr=10.5, te_sdr=10.0 after 300k=500k
# + 23a_4gpus.yaml  the same as 22a_4gpus. layer=6, 0.5dB lower than 22a_4gpus
# + 24a_4gpus.yaml  the same as 22a_4gpus, layer=24, bs=4, te=10.3 after 1M steps. Overlap_te=10.7
# 25a_4gpus.yaml	the same as 22a_4gpus, volume aug, fast_tr=10.2, te=10.35, overlap_te=10.65 after 900k steps
# 26a_4gpus.yaml	the same as 22a_4gpus, pitch aug, fast_tr=9.0, te=9.7 after 900 steps.
# 27a_4gpus.yaml    BSRoformerLinearBand, layer=12, t=201, f=61, patch=(4,1), gpu=4, bs=8, not good.
# 28a_4gpus.yaml    BSRoformerMelLinear, layer=6, t=201, f=120, patch=(4,1), gpu=4, bs=8, 0.5 dB lower than 22a_
# 29a_4gpus.yaml    full Transformer, others same as 22a
# 29b_4gpus.yaml    full Transformer, others same as 22a, 24 layers
# 30a_4gpus.yaml	BsRoformer new RoPE, others same as 22a