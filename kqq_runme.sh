CUDA_VISIBLE_DEVICES=1 python train.py --model_name=UNet

CUDA_VISIBLE_DEVICES=1 python train.py --model_name=BSRoformer \
	--clip_duration=4.0 \
	--batch_size=4 \
	--lr=3e-4

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes 4 train_accelerate.py --model_name=BSRoformer4a --clip_duration=4.0 --batch_size=6 --lr=3e-4

CUDA_VISIBLE_DEVICES=1 python evaluate.py \
	--model_name="BSRoformer" \
	--ckpt_path="checkpoints/train_accelerate/BSRoformer/step=390000.pth" \
	--clip_duration=2. \
	--batch_size=16

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --multi_gpu --num_processes 8 train_accelerate.py --model_name=BSRoformer2 --clip_duration=2.0 --batch_size=3 --lr=3e-4


CUDA_VISIBLE_DEVICES=5 python tmp_ft.py --model_name=BSRoformer15a    --clip_duration=1.0     --batch_size=4  --lr=3e-4 --ckpt_path="checkpoints/train_accelerate_bf16/BSRoformer15a/latest_ema.pth"

CUDA_VISIBLE_DEVICES=1,2,3,5 accelerate launch --multi_gpu --num_processes 4 ft_accelerate_bf16.py --model_name=BSRoformer15a --clip_duration=3.0 --batch_size=2 --lr=3e-4 --ckpt_path="checkpoints/train_accelerate_bf16/BSRoformer15a/step=440000_ema.pth"

CUDA_VISIBLE_DEVICES=0 python inference.py \
	--model_name="BSRoformer" \
	--ckpt_path="checkpoints/train_accelerate/BSRoformer/step=390000.pth" \
	--clip_duration=2.0 \
	--batch_size=2


####
CUDA_VISIBLE_DEVICES=1 python train_enc_dec.py --model_name=EncDec --clip_duration=4.0 --batch_size=4 --lr=3e-4


# BSRoformer3	window=8192
# BSRoformer4a	mag, phase
# BSRoformer4b	dct
# BSRoformer5	fixed
# BSRoformer6a	full 2d, not so good on test
# + BSRoformer7a	bs, only 1 early fc, bs=4, clip=4s, 10.05 dB
# BSRoformer8a	combine attention, L=24, overfit a bit
# BSRoformer8b	combine attention x 3,
# BSRoformer10a	L=24, clip=2, bs=3
# + BSRoformer10b	mel, L=12, clip=4, bs=4, ema=10.29 dB
# BSRoformer11a	L=12, clip=4, bs=4, combine att, not run
# BSRoformer12a	L=12, clip=4, bs=4, combine att x 3, ema=9.6 dB
# BSRoformer13a	mel, L=12, clip=4, bs=4, others same as 10a, ema=
# BSRoformer14a	mel, L=12, clip=4, bs=4, full Transformer, ema=8.2 dB
# + BSRoformer15a	10b, mel, L=12, patch=(1, 4), clip=3, bs=2, ema=10.9 dB
# + BSRoformer16a	10b, mel, L=12, patch=(1, 1), clip=3, bs=1, ema=
# BSRoformer17a	Transformer-UNet, clip=4, bs=2, ema=
