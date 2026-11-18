import torch
import os
from pathlib import Path
import h5py
import json
import re
import math
import numpy as np
from scipy.optimize import fsolve, root_scalar, newton
from torch import Tensor
from mss.utils import fast_sdr
import matplotlib.pyplot as plt
import librosa
import soundfile
import time
from statistics import mode
# from torchvision.io import read_video, write_video

import numpy as np
from mss.models2.dsp5.banks import erb_to_hz, hz_to_erb


def add():
    from transformers import T5Tokenizer, T5EncoderModel

    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    encoder = T5EncoderModel.from_pretrained("t5-base")

    text = [
        "Task: video generation. Scene: a dog running on the beach."
    ]

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    with torch.no_grad():
        outputs = encoder(**inputs)

    text_embeds = outputs.last_hidden_state
    # shape: [B, L, 768]

    from IPython import embed; embed(using=False); os._exit(0)


def add2():

    import torch
    from transformers import AutoTokenizer, AutoModel
    import torchaudio

    device = "cuda"

    model = AutoModel.from_pretrained("laion/clap-htsat-unfused")
    tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")
    model.to(device)

    # ---- 文本部分 ----
    texts = [
        "the sound of a cat", 
        "a dog barking loudly"
    ]
    text_inputs = tokenizer(texts, padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        latent = model.get_text_features(**text_inputs)  # (b, d)

    from IPython import embed; embed(using=False); os._exit(0)


def add3():

    import torch
    from transformers import CLIPTokenizer, CLIPTextModel

    device = "cuda"

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    # text_encoder.eval()  # frozen, 如果只是 inference

    texts = [
        "A dog running on the beach",
        "A cat sitting on a sofa"
    ]

    tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    tokens = {k: v.to(device) for k, v in tokens.items()}

    with torch.no_grad():
        latent = text_encoder(**tokens).last_hidden_state  # [B, T, D]

    sentence_embeddings = latent.mean(dim=1)  # [B, D]


def add4():

    vae_dir = "datasets/gtzan_vae"

    labels = sorted(os.listdir(Path(vae_dir, "genres")))

    metas = []

    for k, label in enumerate(labels):
        
        print("{}/{}".format(k, len(labels)))

        paths = sorted(list(Path(vae_dir, "genres", label).glob("*.h5")))

        for path in paths:

            with h5py.File(path, 'r') as hf:
                fps = hf.attrs["fps"]
                n_frames = hf["latent"].shape[0]
                duration = n_frames / fps

            meta = {
                "task": "text_to_music",
                "dataset": "gtzan",
                "input": {
                    "text": {
                        "prompt": label,
                        "language": "en"
                    }
                },
                "target": {
                    "audio": {
                        "path": str(path),
                        "format": "levo_vae", 
                        "fps": fps,
                        "num_frames": n_frames,
                        "duration": duration
                    }
                }
            }
            metas.append(meta)
           
    out_path = "_zz.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for meta in metas:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    print(f"Write out to {out_path}")


def add5():
    from ttstokenizer import TTSTokenizer

    # tokenizer = TTSTokenizer()
    # print(tokenizer("Text to tokenize"))

    from ttstokenizer import IPATokenizer

    tokenizer = IPATokenizer()
    print(tokenizer("Text to tokenize"))
    text = "Text to tokenize"

    from IPython import embed; embed(using=False); os._exit(0)
    tokens = tokenizer.super().__call__(text)

    vocab = tokenizer.ipavocab()

    # Map ARPABET tokens to IPA tokens and join as a string

    text = "".join(vocab.get(x, x) for x in tokens)
    
    
    # IPA token lookup
    tokens = tokenizer.ipatokens()

    # Lookup token ids and return
    a1 =  np.array([tokens.get(x, x) for x in text], dtype=np.int64) if self.tokenize else text

    from IPython import embed; embed(using=False); os._exit(0)


VOCAB = [
    "<pad>", "<bos>", "<eos>", "<sil>",

    # vowels
    "i", "ɪ", "e", "ɛ", "æ", "ɑ", "ɔ", "o", "ʊ", "u",
    "ʌ", "ə", "ɜ", "ɚ",

    # consonants
    "p", "b", "t", "d", "k", "g",
    "f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "h",
    "m", "n", "ŋ",
    "l", "r", "j", "w",
    "tʃ", "dʒ",

    # suprasegmentals
    "ˈ", "ˌ", "ː", " "
]

phoneme2id = {p: i for i, p in enumerate(VOCAB)}
id2phoneme = {i: p for i, p in enumerate(VOCAB)}  


def add6():
    from phonemizer import phonemize

    phones = phonemize(
        ["Hello world!", "ds few"],
        language="en-us",
        backend="espeak",
        strip=True,
        preserve_punctuation=False,
        with_stress=True
    )
    print(phones)
    # ids = [phoneme2id[p] for p in phones]
    # print(ids)
    # a1 = [id2phoneme[id] for id in ids]
    # print(a1)
    from IPython import embed; embed(using=False); os._exit(0)


def normalize_text(x: str) -> str:
    x = re.sub(r"[^\w\s]", " ", x.lower())  # Remain char and digit only
    x = re.sub(r"\s+", " ", x)  # Remove extra spaces
    return x.strip()


def add7():
    a1 = normalize_text("asdf ewio',asdfpi jiavem.")
    print(a1)


def add8():

    from transformers import CLIPProcessor, CLIPModel

    device = "cuda"

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # image = Image.open("image.jpg")
    # image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    image = [np.random.randint(0, 255, (480, 720, 3), dtype=np.uint8) for _ in range(4)]

    inputs = processor(
        text=["a dog", "a cat"],
        images=image,
        return_tensors="pt",
        padding=True
    ).to(device)

    outputs = model(**inputs)
    logits = outputs.logits_per_image
    probs = logits.softmax(dim=1)

    print(probs)
    from IPython import embed; embed(using=False); os._exit(0)


def add9():

    from transformers import CLIPProcessor, CLIPModel

    device = "cuda"

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # image = Image.open("image.jpg")
    # image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    path = "/datasets/AVE/AVE_Dataset/AVE/DTWkL01Z8So.mp4"
    video, _, info = read_video(path, output_format="TCHW", pts_unit="sec")
    from IPython import embed; embed(using=False); os._exit(0)

    inputs = processor(
        text=["a dog", "a cat"],
        images=image,
        return_tensors="pt",
        padding=True
    ).to(device)

    outputs = model(**inputs)
    logits = outputs.logits_per_image
    probs = logits.softmax(dim=1)

    print(probs)


def add10():
    path = "/datasets/AVE/AVE_Dataset/AVE/DTWkL01Z8So.mp4"
    os.system(f'ffmpeg -i {path} -vf "minterpolate=fps=25" _tmp.mp4') 


def add11():

    from torch import Tensor
    from audio_flow.encoders.audio.levo_vae import LevoVAE
    device = "cpu"
    vae = LevoVAE().to(device)
    vae.eval()

    x = Tensor(np.zeros((1, 2, 96000*2))).to(device)
    y = vae.encode(x)

    z = vae.decode(y)

    z = vae.decode(torch.zeros_like(y))

    from IPython import embed; embed(using=False); os._exit(0)


def add12():

    from audio_flow.utils import load_jsonl
    from collections import Counter

    path = "/home/qiuqiangkong/my_code_202308-/audio_flow4/jsonls/tts/train/libritts_train-other-500.jsonl"
    metas = load_jsonl(path)

    strs = [meta["input"]["text"]["content"] for meta in metas]

    # a1 = list(set(list("".join(strs).lower())))
    char_count = Counter("".join(strs).lower())

    from IPython import embed; embed(using=False); os._exit(0)

def add13():

    from audio_flow.utils import load_jsonl
    from audio_flow.encoders.text.char import VOCAB

    path = "/home/qiuqiangkong/my_code_202308-/audio_flow4/jsonls/tts/train/libritts_train-clean-100.jsonl"
    metas = load_jsonl(path)
    tmp = []
    for meta in metas:
        duration = meta["target"]["audio"]["duration"]
        content = normalize_text(meta["input"]["text"]["content"], VOCAB)
        n_chars = len(content)
        tmp.append(n_chars / duration)
    print(np.mean(tmp))
    from IPython import embed; embed(using=False); os._exit(0)


def normalize_text(text: str, vocab) -> str:
        return "".join(c for c in text.lower() if c in vocab)


def add14():

    import whisper

    # 加载模型（可选: tiny, base, small, medium, large）
    model = whisper.load_model("large")

    # 识别音频
    path = "./results/train3_trunc/tts_ljspeech_07b/steps=700000_ema/test,idx=0,prompt=Shortly before the day fixed for execution, Bishop made a full confession, the bulk of which bore the impress of truth,,gen.wav"
    result = model.transcribe(path)

    print(result["text"])


def add15():

    from datasets import load_dataset
    captions = load_dataset("disco-eth/jamendo-fma-captions")

    captions["train"][0]
    from IPython import embed; embed(using=False); os._exit(0)


def add16():

    import torch
    import librosa
    from transformers import (
        Qwen2_5OmniForConditionalGeneration,
        AutoProcessor,
        BitsAndBytesConfig
    )

    model_name = "Qwen/Qwen2.5-Omni-3B"

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=quant_config
    )

    processor = AutoProcessor.from_pretrained(model_name)

    audio, sr = librosa.load("../mss2/assets/music_10s.wav", sr=16000)

    audio = audio[:5 * sr]

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # 推理
    outputs = model.generate(**inputs, max_new_tokens=128)

    from IPython import embed; embed(using=False); os._exit(0)
    result = processor.batch_decode(outputs, skip_special_tokens=True)

    print(result)


def add16b():
    import torch
    import librosa
    from transformers import (
        Qwen2_5OmniForConditionalGeneration,
        AutoProcessor,
        BitsAndBytesConfig
    )

    model_name = "Qwen/Qwen2.5-Omni-3B"

    # 4bit 量化
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=quant
    )

    processor = AutoProcessor.from_pretrained(model_name)

    # 读取音频
    audio, sr = librosa.load("../mss2/assets/music_10s.wav", sr=16000)

    # 建议限制长度
    audio = audio[:1 * sr]
    # audio = audio[:5 * sr]

    inputs = processor(
        text="Describe the music.",
        audios=audio,
        sampling_rate=sr,
        return_tensors="pt"
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        use_cache=False
    )

    caption = processor.batch_decode(outputs, skip_special_tokens=True)

    print(caption)
    from IPython import embed; embed(using=False); os._exit(0)


def add17():

    import nemo.collections.speechlm2 as slm

    model = slm.models.SALM.from_pretrained("path/to/checkpoint").eval()

    audio_signal, sr = torchaudio.load("music.wav")
    audio_signal = audio_signal.to(model.device)

    prompt = [{"role":"user","content":model.audio_locator_tag}]
    output = model.generate(prompts=[prompt], audios=audio_signal, audio_lens=[audio_signal.shape[-1]])
    print(model.tokenizer.ids_to_text(output[0]))


def add19():

    import torch
    import librosa
    from transformers import AutoProcessor, AutoModelForCausalLM

    # 选用 LTU-small 这个音频 caption 模型
    model_name = "LTU-lang/ltu-small-cap-en"

    # 加载模型 & processor
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # 读取音频
    audio, sr = librosa.load("../mss2/assets/music_10s.wav", sr=16000)

    # 限制音频长度（建议 <= 8s）
    audio = audio[:8 * sr]

    # 构造输入
    inputs = processor(
        text="Describe the music.",
        audios=audio,
        sampling_rate=sr,
        return_tensors="pt"
    )

    # 把输入移到 GPU
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # 推理生成 caption
    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        use_cache=False
    )

    # 解码输出
    caption = processor.batch_decode(outputs, skip_special_tokens=True)

    print("Generated Caption:", caption)


def add19():

    import soundfile
    from audio_flow.encoders.audio.levo_vae import LevoVAE

    device = "cuda"

    if False:
        z = torch.zeros((64, 250)).to(device)
    else:
        import pickle
        z = pickle.load(open("_zz.pkl", "rb"))
        z = torch.Tensor(z).to(device)

    vae = LevoVAE().to(device)
    y = vae.decode(z).data.cpu().numpy()[0]  # (c, l)
    soundfile.write(file="_zz.wav", data=y[0], samplerate=48000)

    from IPython import embed; embed(using=False); os._exit(0)


def add20():

    from mss.models2.dsp3.banks import erb_linear_banks
    sr = 48000
    n_bands = 128
    max_bandwidth = 800
    banks = erb_linear_banks(sr=sr, n_bands=n_bands, max_bandwidth=max_bandwidth)
    print(banks)


def add21():

    from mss.models2.dsp3.banks import erb_linear_banks_triangle
    from mss.models2.dsp3.subband_fast_triangle import SubbandFilter

    sr = 48000
    n_bands = 112
    max_half_bandwidth = 390
    factor = 800
    chunk_size = 16
    banks = erb_linear_banks_triangle(sr=sr, n_bands=n_bands, max_half_bandwidth=max_half_bandwidth)
    # print(banks)

    sb_filter = SubbandFilter(sr, banks, factor, chunk_size=chunk_size).to(device)

    from IPython import embed; embed(using=False); os._exit(0)


def add22():

    from mss.models2.dsp3.banks import erb_linear_banks
    from mss.models2.dsp3.subband_fast import SubbandFilter

    sr = 48000
    n_bands = 112
    max_bandwidth = 390
    factor = sr // 400
    chunk_size = 16
    device = "cuda"

    banks = erb_linear_banks(sr=sr, n_bands=n_bands, max_bandwidth=max_bandwidth)
    sb_filter = SubbandFilter(sr, banks, factor, chunk_size=chunk_size).to(device)
    print(banks)
    bandwidths = [bank[1] - bank[0] for bank in banks]

    rs = np.random.RandomState(1234)
    audio = rs.uniform(low=-1, high=1, size=(4, 2, sr * 2))
    audio = Tensor(audio).to(device)  # (c, l)
            
    # Analysis
    x = sb_filter.analysis(audio)  # (b, c, k, l)
    y = sb_filter.synthesis(x)
    sdr = fast_sdr(audio.cpu().numpy(), y.cpu().numpy())
    print(sdr)


    from IPython import embed; embed(using=False); os._exit(0)


def add23():

    sr = 48000
    n_bands = 128
    max_half_bandwidth = 390
    chunk_size = 16  # Try to tune this to balance RAM and computation speed
    factor = sr // 800

    from mss.models2.dsp3.banks import exp_linear_banks
    banks = exp_linear_banks(sr=sr, n_bands=n_bands)
    bandwidths = [bank[-1] - bank[0] for bank in banks]

    for i in range(len(banks)):
        print(i, banks[i], bandwidths[i])


def add24():

    r = 2
    rs = []
    total = 0
    for i in range(70):
        rs.append(r)
        total += r
        r = r ** 1.03

    print(rs)
    print(total)


def add25():

    n_bands = 128
    sr = 48000

    max_bw = 390
    # q = 2**(1/12)
    q = 1.08

    a1 = (390 * q) / (q - 1) - sr / 2 + n_bands * max_bw
    a1 = a1 / 390 - 1
    exp = q ** a1
    f0 = max_bw / (q - 1) / exp

    n = 1 + math.log((max_bw / f0) / (q - 1)) / math.log(q)
    n = math.ceil(n)

    # Calibriate f0
    f0 = max_bw / (q ** n - q ** (n-1))

    freqs = []
    for i in range(n+1):
        freqs.append(f0 * q**i)

    n_linear_bands = n_bands - n
    bw = (sr / 2 - f0 * q ** n) / n_linear_bands
    for i in range(n_linear_bands):
        freqs.append(freqs[-1] + bw)

    freqs[-1] = sr / 2

    plt.plot(freqs)
    plt.savefig("_zz.pdf")

    from IPython import embed; embed(using=False); os._exit(0)


def add26():
    from mss.models2.dsp3.banks import mel_linear_banks_triangle, erb_linear_banks_triangle, exp_linear_banks2, erb_linear_ex_banks_triangle, linear_banks_triangle

    sr = 48000
    n_bands = 128
    max_bandwidth = 390

    lines = []

    banks = mel_linear_banks_triangle(sr, 118, max_bandwidth)
    freqs = [bank[0] for bank in banks]
    line, = plt.plot(freqs, label="mel_linear")
    lines.append(line)

    banks = erb_linear_banks_triangle(sr, 112, max_bandwidth)
    freqs = [bank[0] for bank in banks]
    line, = plt.plot(freqs, label="erb_linear_banks_triangle")
    lines.append(line)

    # banks = erb_linear_ex_banks_triangle(sr, 112, max_bandwidth, a=21.4, b=0.01)
    banks = erb_linear_ex_banks_triangle(sr, 103, max_bandwidth, a=21.4, b=0.02)
    freqs = [bank[0] for bank in banks]
    line, = plt.plot(freqs, label="erb_linear_ex_banks_triangle, b=0.02")
    lines.append(line)

    banks = erb_linear_ex_banks_triangle(sr, 120, max_bandwidth, a=21.4, b=0.001)
    freqs = [bank[0] for bank in banks]
    line, = plt.plot(freqs, label="erb_linear_ex_banks_triangle, b=0.001")
    lines.append(line)

    banks = erb_linear_ex_banks_triangle(sr, 124, max_bandwidth, a=21.4, b=0.0005)
    freqs = [bank[0] for bank in banks]
    line, = plt.plot(freqs, label="erb_linear_ex_banks_triangle, b=0.0005")
    lines.append(line)

    banks = erb_linear_ex_banks_triangle(sr, 127, max_bandwidth, a=21.4, b=0.0001)
    freqs = [bank[0] for bank in banks]
    line, = plt.plot(freqs, label="erb_linear_ex_banks_triangle, b=0.0001")
    lines.append(line)

    banks = linear_banks_triangle(sr, 128)
    freqs = [bank[0] for bank in banks]
    line, = plt.plot(freqs, label="linear_band")
    lines.append(line)
    # from IPython import embed; embed(using=False); os._exit(0)

    # bw = [bank[1] - bank[0] for bank in banks]
    # for i in range(len(banks)):
    #     print(i, banks[i][0], bw[i])

    banks = exp_linear_banks2(sr, n_bands, max_bandwidth, q=1.05)
    freqs = [bank[0] for bank in banks]
    line, = plt.plot(freqs, label="exp_linear_banks2, q=1.05")
    lines.append(line)

    banks = exp_linear_banks2(sr, n_bands, max_bandwidth, q=1.08)
    freqs = [bank[0] for bank in banks]
    line, = plt.plot(freqs, label="exp_linear_banks2, q=1.08")
    lines.append(line)

    banks = exp_linear_banks2(sr, n_bands, max_bandwidth, q=1.2)
    freqs = [bank[0] for bank in banks]
    line, = plt.plot(freqs, label="exp_linear_banks2, q=1.2")
    lines.append(line)

    plt.legend(handles=lines)
    plt.savefig("_zz.pdf")

    

    from IPython import embed; embed(using=False); os._exit(0)


def add26b():
    from mss.models2.dsp3.banks import mel_linear_banks_triangle, erb_linear_banks_triangle, exp_linear_banks2, erb_linear_ex_banks_triangle, linear_banks_triangle

    sr = 48000
    n_bands = 128
    max_bandwidth = 390

    lines = []

    # banks = mel_linear_banks_triangle(sr, 118, max_bandwidth)
    # freqs = [bank[0] for bank in banks]
    # line, = plt.plot(freqs, label="mel_linear")
    # lines.append(line)

    # banks = erb_linear_banks_triangle(sr, 112, max_bandwidth)
    # freqs = [bank[0] for bank in banks]
    # line, = plt.plot(freqs, label="erb_linear_banks_triangle")
    # lines.append(line)

    a, b = 21.4, 0.001
    banks = erb_linear_ex_banks_triangle(sr, 120, max_bandwidth, a=a, b=b)
    freqs = [bank[0] for bank in banks]
    line, = plt.plot(freqs, label=f"erb_linear_ex_banks_triangle, a={a}, b={b}")
    lines.append(line)

    a, b = 100, 0.001
    banks = erb_linear_ex_banks_triangle(sr, 120, max_bandwidth, a=a, b=b)
    freqs = [bank[0] for bank in banks]
    line, = plt.plot(freqs, label=f"erb_linear_ex_banks_triangle, a={a}, b={b}")
    lines.append(line)

    plt.legend(handles=lines)
    plt.savefig("_zz.pdf")

    

    # from IPython import embed; embed(using=False); os._exit(0)


def add27():
    from mss.models2.dsp3.banks import mel_linear_banks_triangle

    sr = 48000
    n_bands = 64
    max_bandwidth = 790

    lines = []

    banks = mel_linear_banks_triangle(sr, 58, max_bandwidth)
    
    from IPython import embed; embed(using=False); os._exit(0)

    # 64 bins
    self.n_fft = 64
    self.hop_length = 16
    self.patch_size_t = 4
    max_half_bandwidth = 790
    factor = sample_rate // 1600
    chunk_size = 16

    # Subband filter
    banks = mel_linear_banks_triangle(sample_rate, 58, max_half_bandwidth)
    self.sb_filter = SubbandFilter(sample_rate, banks, factor, chunk_size=chunk_size)


def add28():

    path = "assets/music_10s.wav"
    audio, fs = librosa.load(path=path, sr=None, mono=True)

    out = audio.copy().astype(np.float16)
    sdr = fast_sdr(audio, out)
    from IPython import embed; embed(using=False); os._exit(0)


def add29():
    from mss.augmentations.numpy.resample import RandomResample
    from mss.augmentations.numpy.pitch import RandomPitch

    sr = 48000
    path = "assets/music_10s.wav"
    audio, fs = librosa.load(path=path, sr=sr, mono=True)
    audio = audio[None, :]

    # from IPython import embed; embed(using=False); os._exit(0)
    # aug = RandomResample(sr=sr, min_ratio=0.95, max_ratio=0.95)
    aug = RandomPitch(sr=sr, min_step=4.0, max_step=4.0)
    while True:
        t1 = time.time()
        y = aug(audio)
        print(time.time() - t1)
    soundfile.write(file="_zz.wav", data=y.T, samplerate=sr)



def add30():
    sr = 48000
    n_fft = 2048
    path = "assets/music_10s.wav"
    audio, fs = librosa.load(path=path, sr=sr, mono=True)
    audio = Tensor(audio)
    X = torch.stft(
        input=audio, 
        n_fft=n_fft,
        hop_length=480,
        window=torch.hann_window(n_fft),
        normalized=True,
        onesided=False,
        return_complex=True
    ).T.abs()[:, 0 : n_fft // 2 + 1]
    X2 = librosa.core.stft(y=audio.n, n_fft=2048, hop_length=480, window='hann', center=True).abs()

    plt.matshow(np.log10(X).T, origin='lower', aspect='auto', cmap='jet')
    plt.savefig("_zz.pdf")
    from IPython import embed; embed(using=False); os._exit(0)


def add30b():
    sr = 48000
    n_fft = 2048
    path = "assets/music_10s.wav"
    audio, fs = librosa.load(path=path, sr=sr, mono=True)

    X = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=480, n_mels=512).T
    plt.matshow(np.log10(X).T, origin='lower', aspect='auto', cmap='jet')
    plt.savefig("_zz.pdf")
    from IPython import embed; embed(using=False); os._exit(0)


def add31():
    # x = np.arange(0, 1, 0.01)

    # a = 100
    # y = np.log10(1 + a * x)

    x = torch.arange(0, 3, 0.01)
    mu = 255
    y = torch.sign(x) * torch.log1p(mu * torch.abs(x)) / math.log(1.0 + mu)

    print(y)


def add32():
    from mss.models2.dsp3.banks import mel_linear_banks_triangle
    from mss.models2.dsp3.subband_fast_triangle import SubbandFilter

    sr = 48000
    n_bins = 32

    if n_bins == 32:
        n_bands = 28
        max_half_bandwidth = 1590
        chunk_size = 16  # Try to tune this to balance RAM and computation speed
        factor = sr // (1600 * 2)  # Can be smaller but not larger!
        device = "cuda"

    elif n_bins == 64:
        n_bands = 58
        max_half_bandwidth = 790
        chunk_size = 16  # Try to tune this to balance RAM and computation speed
        factor = sr // (800 * 2)  # Can be smaller but not larger!
        device = "cuda"

    elif n_bins == 128:
        n_bands = 118
        max_half_bandwidth = 390
        chunk_size = 16  # Try to tune this to balance RAM and computation speed
        factor = sr // (400 * 2)  # Can be smaller but not larger!
        device = "cuda"

    elif n_bins == 256:
        n_bands = 235
        max_half_bandwidth = 190
        chunk_size = 16  # Try to tune this to balance RAM and computation speed
        factor = sr // (200 * 2)  # Can be smaller but not larger!
        device = "cuda"

    # Melbanks
    banks = mel_linear_banks_triangle(sr, n_bands, max_half_bandwidth)
    sb_filter = SubbandFilter(sr, banks, factor, chunk_size=chunk_size).to(device)
    print(len(banks))

    for _ in range(5):

        # Audio
        rs = np.random.RandomState(1234)
        audio = rs.uniform(low=-1, high=1, size=(4, 2, sr * 2))
        audio = Tensor(audio).to(device)  # (c, l)
                
        # Analysis
        t0 = time.time()
        x = sb_filter.analysis(audio)  # (b, c, k, l)
        y = sb_filter.synthesis(x)
        
        # Print
        t1 = time.time() - t0
        sdr = fast_sdr(audio.cpu().numpy(), y.cpu().numpy())
        print(f"time: {t1:.4f} s, latent: {x.shape}, SDR: {sdr:.2f} dB")




def add33():

    from mss.models2.dsp5.banks import refine_erb_bands

    n_bands = 128
    sr = 48000
    bw = 48

    freqs, idxes = refine_erb_bands(n_bands, sr, bw)

    from IPython import embed; embed(using=False); os._exit(0)



if __name__ == '__main__':
    add33()