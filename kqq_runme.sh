
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

CUDA_VISIBLE_DEVICES=7 python evaluate.py \
    --config="./kqq_configs/115d_drums.yaml" \
    --ckpt_path="./checkpoints/train2/115d_drums/step=150000_ema.pth" \
    --out_dir="_tmp/drums"
    
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
# 56b.yaml      1122444 overlap band
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
# 64a.yaml      UTransformerCat, others same as 42a
# 65a.yaml      abs pos, slightly worse
# 65b.yaml      abs pos, all layers
# + 65c.yaml      layer scale. Better than 42a
# 65d.yaml      layer scale, all freq
# 66a.yaml      stft 128, 512, 2048 cat
# 66a2.yaml      stft 128, 512, 2048 cat, weight, worse than 66a
# 67a.yaml      1D transformer
# 67b.yaml      BSRoformer 2D rope, pretrain
# 67b2.yaml     BSRoformer 2D rope, finetune
# 68a.yaml      BS + conv1d_hop30
# 68b.yaml      BS + conv1d_hop_5_3_2
# 68b2.yaml      BS + conv1d_hop_5_3_2, larger kernel

# 80a.yaml      frame theory, stft+wave
# 80b.yaml      frame theory, stft+haar

# 81a.yaml      filterband, 65 bands, 2s=1200samples, others same as 42a
# + 81b.yaml      filterband, 69 bands, 2s=800samples, others same as 42a
# 81c.yaml      2x, 4x, 8x, 16x downsample, others same as 81c
# 82a.yaml      wav, others same as 81b. Not work
# 83a.yaml      linear banks, others same as 81b
# 83b.yaml      melbanks2, others same as 81b
# + 83c.yaml      erb band, others same as 81b
# 84a.yaml      erb band, wav, others same as 83c
# 85a.yaml      mel subband 256, patch=(4, 4)
# 85b.yaml      erb subband 256, patch=(4, 4)
# 86a.yaml      mel subband 64, patch=(4, 1), fft=64, hop=8
# 86b.yaml      mel subband 64, patch=(1, 1), fft=64, hop=32
# + 87a.yaml      erb subband 64, patch=(4, 1), fft=64, hop=8, erb better than mel
# 87b.yaml      erb subband 64, patch=(1, 1), fft=64, hop=32
# 88a.yaml      erb subband 256, patch=(4, 4), ds=240, fft=16, hop=2
# 88b.yaml      erb subband 256, patch=(4, 4), ds=240, fft=64, hop=2
# 89a.yaml      erb subband 32, patch=(4, 1), ds=30, fft=128, hop=16
# 89b.yaml      erb subband 128, patch=(4, 2), ds=120, fft=32, hop=4
# + 89c.yaml      erb subband 128, patch=(4, 1), ds=120, fft=32, hop=4
# + 89c2.yaml      erb subband 128, patch=(4, 1), ds=120, fft=32, hop=4
# + 89d.yaml      erb subband 256, patch=(4, 1), ds=240, fft=16, hop=2
# + 90a.yaml      fft=32, hop=8, others same as 87a, sdr=9.0dB
# 90b.yaml      fft=128, hop=8, others same as 87a, sdr=8.0dB
# 90c.yaml      fft=16, hop=8, others same as 87a
# 90d.yaml      fft=16, hop=8, rectangle window, others same as 87a

# 91a.yaml      conv1d, others same as 87a, sdr=8.7dB
# 92a.yaml      erb subband 64, patch=(4, 1), multi nfft/hop, others same as 87a
# 93a.yaml      multil stft, others same as 87a
# 94a.yaml      nfft=16, hop=2, patch=(16, 1), others same as 87a, slightly better than 87a
# 95a.yaml      subband 1D Transformer, erb subband=64, patch=(4, 1) others same as 87a
# 95b.yaml      erb subband=64, n_fft=16, hop=4, patch=(1, 1), others same as 87a
# 96a.yaml      subband=128, n_fft=16, hop=4, patch=(1, 1), others same as 87a
# 97a.yaml      conv1d, kernel=32, others as 87a
# + 98a.yaml      overlap subband, others similar to 89c2, but not the same

# 99a.yaml      sp loss, others same as 87a
# 99a2.yaml     sp loss scale, others same as 87a, good.
# 99b.yaml      sp + wav loss, others same as 87a
# 99c.yaml      old loss same same as 87a
# 99d.yaml      sp loss, hop=147 same same as 87a
# 99e.yaml      same same as 87a
# 99e2.yaml     win=2048, others same same as 87a
# 99f.yaml      l1 loss
# + 99g.yaml    sp loss, 2048, hop=512, SDR=8.9dB
# 99g2.yaml     sp loss multi, 256, 512, 1024, 2048, 4096, SDR=8.5dB
# x 99h.yaml      logsp loss, others same as 87a 
# 99i.yaml      subband stft loss, others same as 87a 

# 100a.yaml     mel bandsplit, others same as 89c2
# + 101a.yaml     swiGLU, others same as 89c2
# - 102a.yaml     only remain 1/6 att, others same as 89c2
# 102b.yaml     only remain 1/2 att, others same as 89c2
# - 103a.yaml     scale, others same as 89c2
# 103b.yaml     scale, constant, others same as 89c2
# 103c.yaml     scale, 2d, others same as 89c2
# 103d.yaml     scale, 2d, decompose, others same as 89c2
# - 104a.yaml     pool attention, others same as 89c2
# 105a.yaml     24 layers, others same as 89c2
# 106a.yaml     cross shape att, others same as 89c2 (bs 26s, noatt 13s->20s, fullatt 120s, cross 34s)
# 107a.yaml     unet-transformer
# + 107a2.yaml    unet-transformer， fix
# 107b.yaml     unet-transformer, pixel
# - 107c.yaml     unet-transformer, pixel_rnn
# 108a.yaml     subbandloss, others same as 89c2
# + 108b.yaml     subband_stft_loss, others same as 89c2
# + 108c.yaml     subband_stft_loss, 1,4,16,64, others same as 89c2
# 109a.yaml     wav conv1d (jimmy), others same as 89c2
# jimmy56b.yaml 
# 110a.yaml     erb band, others same as jimmy56b.yaml
# ++ 111a.yaml     trianle band, others same as 89c2
# (no) 111b.yaml     overlap band, others same as 89c2
# 112a.yaml     lowband fractional stft, others same as 89c2
# 113a.yaml     mel bandsplit unet, others same as 107a2
# 114a.yaml     patch=(2, 1), others same as 89c2
# 114b.yaml     patch=(1, 1), others same as 89c2
# 115a.yaml     exp band, others same as 111a
# 115b.yaml     exp_linear2 band, others same as 111a
# 115c.yaml     erb, b=0.02, others same as 111a
# + 115d.yaml     mel, others same as 111a, good.
# 115e.yaml     erb, b=0.001, others same as 111a
# 115f.yaml     erb, b=0.0005, others same as 111a
# 115g.yaml     erb, b=0.0001, others same as 111a
# 115h.yaml     linear band, others same as 111a
# 115i.yaml     mel + erb + linear bands, others same as 111a
# - 116a.yaml     band wise matrix, others same as 89c2
# 117a.yaml     glu, others same as 115d
# 118a.yaml     glu, unet, others same as 117a
# 119a.yaml     spectral constant loss, others same as 115d
# 120a_bass.yaml    win=128, hop=8, patch=(4, 1), others same as 115d
# 121a_bass.yaml    win=128, hop=32, patch=(1, 1) others same as 115d

# 122a_bass.yaml    train3b, stem aug, pitch aug +-0, others same as 115d
# 122b_bass.yaml    train3b, stem aug, pitch aug +-2, others same as 115d
# 122c_bass.yaml    train3b, stem aug, pitch aug +-4, others same as 115d
# 123a.yaml     log mel loss, others same as 115d

# 124a.yaml     loss_01a, l1 wav loss, others same as 115d, -0.8 dB
# 125a.yaml     loss_02a, sp loss, others same as 115d, -0.4 dB
# + 125b.yaml     same as 125a, grad clip 1.0
# 126a.yaml     loss_03a, multi sp loss, others same as 115d
# 126b.yaml     loss_03a, multi sp loss, hop=147, others same as 115d
# 127a.yaml     loss_04a, log sp loss, others same as 115d, not work.
# 128a.yaml     loss_05a, log(1+x) loss, others same as 115d
# 129a.yaml     loss_06a, sdr loss, win=2048, others same as 115d
# 130a.yaml     loss_07a, wav sdr loss2, win=2048, others same as 125b, train2b
# 131a.yaml     loss_08a, log sp, others same as 125a

# train2b.py  clip norm
# 132a.yaml      loss_09a, sp loss + wav sdr loss, others as 125b
# 132b.yaml      loss_09a, sp loss + wav sdr loss 2s, others as 125b
# 133a.yaml      loss_10a, sp loss + log sp loss, others as 125b
# 134a.yaml      loss_11a, sp loss + mel loss, others as 125b

# train4_01a.yaml      zfturbo aug, others same as 115d

# zf_01a.yaml   bass, others same as 115d

# =============
# musdb18hq_mix.py  mix_src + aug
# musdb18hq_mix2.py  mix_src + stem-wise aug

# =============
# random_song_sampler_mix2.py  mix stem-wise sampler

# =============
# + train2b.py: clip norm
# --- train3.py augmentation ---
# --- train3b.py: stem-wise augmentation ---
# --- train_zfturbo.py
# --- train4.py zfturbo augmentation
# 70.yaml       


# =============
# dsp/dsp3 filter compare
# 

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


# 115d, 150k, vocals: 9.05 bass: 5.96 drums: 6.94 other: 5.66