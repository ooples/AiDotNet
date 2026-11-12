"""Execute pinned, unmodified MelGAN AudioDataset/Audio2Mel bodies for independent goldens.

Requires numpy, scipy, librosa and torch. Only API adapters for current librosa/torch
are applied; the author's float32 waveform, window, bank and magnitude are retained.
"""
import ast
import hashlib
import json
import random
import tempfile
import urllib.request
import wave
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

COMMIT = "6488045bfba1975602288de07a58570c7b4d66ea"
BLOBS = {
    "mel2wav/dataset.py": "0982af8a60f09e5c0e4c9e4e0990e41f2a848832",
    "mel2wav/modules.py": "da39c5d61efcc5593413a3650edc5a57e2dd0319",
}


def author_class(path, name, namespace):
    data = urllib.request.urlopen(
        f"https://raw.githubusercontent.com/descriptinc/melgan-neurips/{COMMIT}/{path}"
    ).read()
    assert hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() == BLOBS[path]
    node = next(n for n in ast.parse(data).body if isinstance(n, ast.ClassDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), namespace)
    return namespace[name]


# The old release calls librosa with positional arguments and torch.stft without
# return_complex. Adapt those API signatures without editing either author body.
def mel_fn(sr, n_fft, n_mels, fmin, fmax):
    return librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)


original_stft = torch.stft
def legacy_stft(*args, **kwargs):
    return original_stft(*args, **kwargs, return_complex=False)
torch.stft = legacy_stft

Audio2Mel = author_class("mel2wav/modules.py", "Audio2Mel",
    dict(torch=torch, nn=nn, F=F, librosa_mel_fn=mel_fn))
AudioDataset = author_class("mel2wav/dataset.py", "AudioDataset",
    dict(torch=torch, F=F, np=np, random=random, Path=Path,
         load=librosa.load, normalize=librosa.util.normalize))
fft = Audio2Mel(n_fft=64, hop_length=16, win_length=64, sampling_rate=4000, n_mel_channels=8)
root = Path(tempfile.mkdtemp(prefix="aidotnet-melgan-recording-reference-"))
result = dict(commit=COMMIT, git_blobs=BLOBS, precision="unmodified released float32", cases=[], features=[])
for name, length in [("short_padding", 37), ("long_sample_crop", 3072), ("silence", 256)]:
    pcm = np.zeros(length, dtype=np.int16) if name == "silence" else np.array(
        [(((i * 37) % 211 - 105) * 128 + ((i % 7) - 3) * 512) for i in range(length)], dtype=np.int16)
    path = root / (name + ".wav")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(4000); wav.writeframes(pcm.tobytes())
    loader = AudioDataset.__new__(AudioDataset)
    loader.sampling_rate = 4000; loader.segment_length = 128; loader.audio_files = [path]; loader.augment = True
    random.seed(17); np.random.seed(17)
    # Record the actual RNG selections; matching the selections in .NET avoids
    # treating different Python/NumPy/.NET RNG algorithms as signal-processing defects.
    amplitude = float(np.random.uniform(0.3, 1.0)); np.random.seed(17)
    start = random.randint(0, length - 128) if length >= 128 else 0; random.seed(17)
    audio = loader[0]
    mel = fft(audio.unsqueeze(0)).detach()
    result["cases"].append(dict(name=name, raw_length=length, segment_samples=128,
        amplitude=amplitude, start_sample=start, maximum_start=max(0, length-128),
        audio=audio.flatten().tolist(), mel=mel.flatten().tolist()))
for scale in [1.0, 1e-5, 0.0]:
    audio = np.array([((i*37)%211-105)/256.0+(i%7-3)/64.0 for i in range(256)], dtype=np.float32) * scale
    result["features"].append(dict(scale=scale, audio=audio.tolist(),
        mel=fft(torch.from_numpy(audio).reshape(1,1,-1)).detach().flatten().tolist()))
destination = Path(__file__).resolve().parents[2] / "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/melgan_recording_reference.json"
destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {len(result['cases'])} recordings and {len(result['features'])} frontend cases to {destination}")
