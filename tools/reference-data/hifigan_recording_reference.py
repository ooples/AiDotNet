"""Independent recording/feature goldens from pinned, unmodified HiFi-GAN loader bodies.

Only librosa keyword and torch.stft return_complex API adapters are supplied.
The author's float32 tensors, bank, window and magnitude epsilon are unchanged.
"""
import ast
import hashlib
import json
import math
import os
import random
import sys
import tempfile
import urllib.request
import wave
from pathlib import Path

import librosa
import numpy as np
import scipy.io.wavfile
import torch

COMMIT = "4769534d45265d52a904b850da5a622601885777"
BLOB = "4502924513ce20fe972732c000b9d81dc6989480"
source = urllib.request.urlopen(f"https://raw.githubusercontent.com/jik876/hifi-gan/{COMMIT}/meldataset.py").read()
assert hashlib.sha1(b"blob " + str(len(source)).encode() + b"\0" + source).hexdigest() == BLOB


def mel_fn(sr, n_fft, n_mels, fmin, fmax):
    return librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)


original_stft = torch.stft
def legacy_stft(*args, **kwargs):
    return original_stft(*args, **kwargs, return_complex=False)
torch.stft = legacy_stft
torch.set_num_threads(1)

namespace = dict(torch=torch, np=np, random=random, math=math, os=os,
    read=scipy.io.wavfile.read, normalize=librosa.util.normalize, librosa_mel_fn=mel_fn,
    MAX_WAV_VALUE=32768.0, mel_basis={}, hann_window={})
names = {"load_wav", "dynamic_range_compression_torch", "spectral_normalize_torch", "mel_spectrogram", "MelDataset"}
nodes = [node for node in ast.parse(source).body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
assert len(nodes) == len(names)
exec(compile(ast.Module(body=nodes, type_ignores=[]), "pinned_hifigan_meldataset.py", "exec"), namespace)
MelDataset = namespace["MelDataset"]
root = Path(tempfile.mkdtemp(prefix="aidotnet-hifigan-recording-reference-"))
result = dict(commit=COMMIT, git_blob=BLOB, precision="released float32", cases=[])
for name, length in [("short_padding", 37), ("exact_segment", 128), ("long_sample_crop", 3072), ("silence", 128)]:
    pcm = np.zeros(length, dtype=np.int16) if name == "silence" else np.array(
        [(((i * 37) % 211 - 105) * 128 + ((i % 7) - 3) * 512) for i in range(length)], dtype=np.int16)
    if name == "long_sample_crop": pcm[0] = 30000
    path = root / (name + ".wav")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(4000); wav.writeframes(pcm.tobytes())
    loader = MelDataset.__new__(MelDataset)
    loader.audio_files = [str(path)]; loader._cache_ref_count = 0; loader.n_cache_reuse = 0
    loader.fine_tuning = False; loader.split = True; loader.segment_size = 128
    loader.n_fft = 64; loader.num_mels = 8; loader.sampling_rate = 4000; loader.hop_size = 16
    loader.win_size = 64; loader.fmin = 0; loader.fmax = 2000; loader.fmax_loss = None
    random.seed(17)
    start = random.randint(0, length - 128) if length >= 128 else 0
    random.seed(17)
    mel, audio, _, loss_mel = loader[0]
    result["cases"].append(dict(name=name, raw_length=length, segment_samples=128, start_sample=start,
        maximum_start=max(0, length - 128), pcm=pcm.tolist(), audio=audio.flatten().tolist(),
        mel=mel.flatten().tolist(), loss_mel=loss_mel.flatten().tolist()))
result["input_mel_basis"] = namespace["mel_basis"]["2000_cpu"].t().flatten().tolist()
result["loss_mel_basis"] = namespace["mel_basis"]["None_cpu"].t().flatten().tolist()
parameter = torch.nn.Parameter(torch.tensor([.25]))
optimizer = torch.optim.AdamW([parameter], lr=2e-4, betas=(.8, .99))
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.9)
result["epoch_learning_rates"] = []
for update in range(6):
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    if (update + 1) % 2 == 0: scheduler.step()
    result["epoch_learning_rates"].append(optimizer.param_groups[0]["lr"])
destination = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    Path(__file__).resolve().parents[2] / "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/hifigan_recording_reference.json")
destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {len(result['cases'])} independent released-loader cases to {destination}")
