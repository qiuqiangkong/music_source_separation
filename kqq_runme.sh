
CUDA_VISIBLE_DEVICES=3 python train.py --config="./kqq_configs/01a.yaml" --no_log
CUDA_VISIBLE_DEVICES=3 python train2_recon.py --config="./kqq_configs_recon/recon_01a.yaml" --no_log

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes 4 train_accelerate.py --config="./kqq_configs/01a.yaml"

CUDA_VISIBLE_DEVICES=0 python inference.py \
    --config="./configs/bsroformer.yaml" \
    --ckpt_path="./checkpoints/tmp_accelerate/01a/step=0_ema.pth" \
    --audio_path="./assets/music_10s.wav" \
    --output_path="./out.wav"

CUDA_VISIBLE_DEVICES=0 python evaluate.py \
    --config="./configs/small.yaml" \
    --ckpt_path="./checkpoints/tmp_accelerate/01a/step=0_ema.pth"
    
# train.py      main
# train2.py     multi classes
# train2_recon.py   reconstruct new dft

# + 01a.yaml  small, strict align, sdr=7.2
# 02a.yaml    group align, same as 01a
# 03a.yaml    mag + phase, same as 01a, no better
# 04a.yaml    mag + phase2, same as 01a, no better
# 04a2.yaml   (in: mag+cmplx), (out: mag=leaky, phase=cmplx), worse
# 04a3.yaml   (in: mag+cmplx), (out: mag=elu, phase=cmplx), slightly worse
# + 04a4.yaml   (in: mag+cmplx), (out: cmplx)
# - 04a5.yaml   (in: cmplx), (out: mag=relu, phase=cmplx), sdr=0dB
# 04a6.yaml   (in: cmplx), (out: mag=leaky, phase=cmplx)
# 04a7.yaml   (in: cmplx), (out: mag=elu, phase=cmplx)
# 04a8.yaml   (in: cmplx), (out: mag=abs, phase=cmplx), sdr=7.2
# + 04a9.yaml   (in: cmplx), (out: cmplx)
# 04b.yaml    mag + phase, white phase, same as 01a
# 04b2.yaml    mag + phase, white phase, same as 01a
# 05a.yaml    patch=(1,4), 4gpus, 1 dB better
# 05b.yaml    patch=(4,1), 4gpus
# + 06a.yaml    stft loss, fast, sdr=7.7
# + 07a.yaml    mul stft, sdr=7.4
# 08a.yaml      volume aug

# --- STFT loss ---
# + 09a.yaml      6 layer, mulstft loss
# 10a.yaml      mag + stft input, others same as 09a, 
# 11a.yaml      wavfeat, others same as 09a
# 12a.yaml      mel fractional fft, others same as 09a

# --- STFT loss, 5 eval ---
# + 13a.yaml      6 layer, mulstft loss
# 13b.yaml      6 layer, mulstft loss, align=group
# 14a.yaml      +mag, others same as 13a
# 14b.yaml      6 layer, mulstft loss, align=group
# 15a.yaml      mel fractional fft2, others same as 13a
# 15b.yaml      bandlinear, others same as 15a
# 16a.yaml      no bandlinear, others same as 13a
# 17a.yaml      uniform bandsplit, others same as 13a
# 17b.yaml      uniform bandsplit overlap, others same as 13a
# 18a.yaml      mel bandsplit2, binary only, others same as 13a
# 19a.yaml      kqq bands (precompute from energy), others same as 13a
# 20a.yaml      fractional fft 4x, others same as 13a
# 20b.yaml      fractional fft 16x, others same as 13a
# 21a.yaml      mel bandsplit3, triangle, bins=256, patch=(4, 4), 7.9dB
# 21b.yaml      mel bandsplit4, triangle, bins=64, patch=(4, 1), 7.9dB
# 22a.yaml      kqq bands (precompute from energy), rectangle, bins=64, patch=(4, 1)
# 22b.yaml      kqq bands (precompute from energy), triangle, bins=64, patch=(4, 1)
# 23a.yaml      mulstft input, others same as 13a
# 24a.yaml      Gabor 4x, others same as 13a

# 25a.yaml      GaborTransform, r=1, for bandsplit
# 25b.yaml      GaborTransform, r=16, for bandsplit
# 26a.yaml      GaborTransform, r=1, orig bandsplit, hop=480
# 26b.yaml      GaborTransform, r=1, orig bandsplit, hop=256
# 26c.yaml      GaborTransform, r=1, orig bandsplit, hop=512
# 26d.yaml      GaborTransform, r=1, for bandsplit, hop=512
# 26e.yaml      GaborTransform, r=1, for bandsplit, hop=480

# 27a.yaml      SparseAttCross, others same as 13a
# 27b.yaml      SparseAttCross3, others same as 13a
# - 28a.yaml      StreamBand, others same as 13a
# 29a.yaml      band init scale, others same as 13a

# 30a.yaml      hop=256, others same as 13a
# 30b.yaml      hop=512, others same as 13a
# 31a.yaml      hop=480, forbandsplit, others same as 13a

# 32a.yaml      newbandsplit, others same as 13a
# 33a.yaml      SparseAttCross+CNN, other same as 27a, worse than 27a
# 34a.yaml      newbatchbandsplit, others same as 13a
# + 35a.yaml      STFT, others same as 26e
# 36a.yaml      newbandsplit2 (nooverlap), others same as 13a
# 36b.yaml      newbatchbandsplit2 (nooverlap), others same as 13a
# 37a.yaml      batchbandsplit, others same as 35a
# 38a.yaml      bandsplit3, others same as 35a
# 38b.yaml      batchbandsplit3, others same as 35a
# 39a.yaml      bandsplit4, rand bias, others same as 35a
# 39a2.yaml     bandsplit5, rand bias, debug=True, others same as 35a, compare to 39a
# 39b.yaml      bandsplit4, zero bias, others same as 35a
# 40a.yaml      same as 35a
# 41a.yaml      bandsplit5, rand bias, others same as 35a, compare to 39a

# + 42a.yaml    batchbandsplit6, new, others same as 13a
# 43a.yaml      GaborTransform, r=16, others same as 42a
# 43b.yaml      GaborTransform, r=16, win=[512, 2048, 4096] others same as 42a
# 43c.yaml      GaborTransform, r=1, win=[512, 2048, 4096] others same as 42a
# 44a.yaml      group, dim=768, others same as 42a
# 44b.yaml      random, dim=768, others same as 42a
# 45a.yaml      fc+sum patch, others same as 42a
# 45b.yaml      split patch, others same as 42a
# + 46a.yaml      patch=(4, 1), others same as 42a
# 46b.yaml      patch=(1, 4), others same as 42a
# 46c.yaml      patch=(1, 1), others same as 42a
# 47a.yaml      full att, others same as 42a
# 48a.yaml      UTransformer, others same as 42a
# 49a.yaml      201x256+Conv2D+Transformer, others same as 42a
# 49b.yaml      201x1025+Conv2D+Transformer, others same as 42a
# 49c.yaml      wav+Conv1D+Transformer, others same as 42a
# 50a.yaml      201x1025, patch=(4, 1), hid=96, result similar to 46c
# 51a.yaml      48a + 201x1025's att
# 52a.yaml      201x256 rectangle att + 42a

# 53a.yaml      Freq patch=(1, 1) + (4, 4)
# - 53b.yaml      Freq patch=(1, 1)blockTransformer + (4, 4)
# 53c.yaml      Sp + Freq patch=(1, 1) + (4, 4)
# 54a.yaml      Melband + linear band
# 54b.yaml      linear band only
# 55a.yaml      shared band, others same as 42a
# 56a.yaml      1122444 band
# 57a.yaml      201x1025, uTransformer

# * 58a.yaml      aug gain, others same as 42a
# * 58b.yaml      aug pitch, others same as 42a
# * 58c.yaml      aug resample, others same as 42a
# * 58d.yaml      aug eq, others same as 42a
# * 58e.yaml      aug resample+time_stretch, others same as 42a
# 59a.yaml      mix multi vocals, others same as 42a
# - 60a.yaml      UTransformer from pixel, others same as 42a, no qk_norm
# 60b.yaml      Test bandsplit60a, otheres same as 42a
# + 61a.yaml      UTransformer from pixel, others same as 42a, no qk_norm
# 62a.yaml      no unet, patch=(4, 4), others same as 61a
# 62b.yaml      no unet, patch=(4, 1), others same as 61a
# 63a.yaml      gabor x4, others same as 46a
# 63b.yaml      gabor x 16, others same as 46a

# train3.py     gpu augmentation

# 70.yaml       

# ====== Reconstruct ======
# recon_01a.yaml    recon_stft_fix, loss=0
# recon_01b.yaml    recon_stft_learnable_dec, loss=0, has bug
# recon_02a.yaml    recon_mel_learnable_dec, 1000bins: tr=40dB, te=20dB. 100bins: 6dB.
# recon_03a.yaml    band_split, 256bands, patch=(4, 4): 45dB.
# recon_03a2.yaml    band_split, 64bands, patch=(4, 4): 40+dB
# recon_03a3.yaml    band_split, 64bands, patch=(4, 16): 40+dB
# recon_03a4.yaml    band_split, 64bands, patch=(4, 64): 40+dB
# recon_04a.yaml    band_split_avg, 
# recon_05a.yaml    band_split_mul_stft
# recon_06a.yaml    band_split_mul_stft



这一部分写的详细些，以limitation开头，自然衔接为什么要做这个，列具体数字
写outcomes, 列具体指标，多少个小时、人数，写具体一点，写once completed, 写 as a contingency plan


Name of the PhD student: Runbang WANG Confirm 2026/27 Prospective Year 1 PhD Student: Yes Name of his/her supervisor: Qiuqiang Kong CV of the PhD student: Attached. The research topic chosen: 基于多模态数据的定位感知算法研究

