"""Run pinned, unmodified PriorGrad dataset, mel and fitted-statistics bodies."""
import ast
import hashlib
import json
import os
import random
import tempfile
import urllib.request
import wave
from pathlib import Path
from types import SimpleNamespace

import librosa
import numpy as np
import torch
from scipy.io.wavfile import read

COMMIT = "8cf4fbcbf451c3affc68ecfa6c970327e927062f"
BLOBS = {
    "PriorGrad-vocoder/dataset.py": "c7d831235f19e804a966bb54cdf25f5c85191b10",
    "PriorGrad-vocoder/preprocess.py": "e129e6ac4bdf66728016f20baaed6cb2b5586c61",
}


def author(path, names, namespace):
    data = urllib.request.urlopen(
        f"https://raw.githubusercontent.com/microsoft/NeuralSpeech/{COMMIT}/{path}"
    ).read()
    assert hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() == BLOBS[path]
    nodes = [n for n in ast.parse(data).body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), namespace)
    return namespace


def mel_fn(sr, n_fft, n_mels, fmin, fmax):
    # Signature adapter only; all returned bank constants retain librosa's float32 dtype.
    return librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)


namespace = author("PriorGrad-vocoder/preprocess.py",
    ["dynamic_range_compression_torch", "spectral_normalize_torch", "get_mel"],
    dict(torch=torch, librosa_mel_fn=mel_fn, mel_basis={}, hann_window={}))
get_mel = namespace["get_mel"]
Dataset = author("PriorGrad-vocoder/dataset.py", ["NumpyDataset"],
    dict(torch=torch, np=np, os=os, random=random, read=read, normalize=librosa.util.normalize,
         MAX_WAV_VALUE=32768.0, get_mel=get_mel, tqdm=lambda x: x))["NumpyDataset"]
params = SimpleNamespace(sample_rate=4000, n_fft=64, n_mels=8, hop_samples=16,
    fmin=0, fmax=2000, crop_mel_frames=8)
root = Path(tempfile.mkdtemp(prefix="aidotnet-priorgrad-recording-reference-"))


def wav(name, length, divisor=1):
    pcm = np.array([(((i*37)%211-105)*128+(i%7-3)*512)//divisor for i in range(length)], dtype=np.int16)
    path = root / (name + ".wav")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1); output.setsampwidth(2); output.setframerate(4000); output.writeframes(pcm.tobytes())
    return path


def loader(path):
    dataset = Dataset.__new__(Dataset)
    dataset.params = params; dataset.filenames = [str(path)]; dataset.is_training = True
    dataset.use_prior = True; dataset.max_energy_override = 4.0
    dataset.energy_min = .02; dataset.energy_max = 4.0; dataset.std_min = .1
    return dataset


result = dict(commit=COMMIT, git_blobs=BLOBS, precision="unmodified released float32", cases=[])
for name, length in [("unaligned_long_recording", 3077), ("exact_crop", 128)]:
    dataset = loader(wav(name, length))
    maximum = length - length % params.hop_samples - params.crop_mel_frames * params.hop_samples
    random.seed(17); start = random.randint(0, maximum); random.seed(17)
    pair = dataset[0]
    result["cases"].append(dict(name=name, raw_length=length, start_sample=start, maximum_start=maximum,
        audio=pair["audio"].tolist(), mel=pair["spectrogram"].T.flatten().tolist(),
        prior_std=pair["target_std"].tolist()))

short = loader(wav("short_removed", 37))
try:
    short[0]
    raise AssertionError("The author loader unexpectedly accepted less than one complete crop.")
except ValueError:
    result["short_recording_rejected"] = True

statistics = loader(wav("loud_statistics", 512))
statistics.filenames.append(str(wav("quiet_statistics", 512, 8)))
statistics.data_root = root
(root / "stats_priorgrad").mkdir()
statistics.compute_stats()
result["statistics"] = dict(raw_length=512, divisors=[1,8],
    minimum=float(np.load(root / "stats_priorgrad/energy_min_train.npy")),
    maximum=float(np.load(root / "stats_priorgrad/energy_max_train.npy")))
destination = Path(__file__).resolve().parents[2] / "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/priorgrad_recording_reference.json"
destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {len(result['cases'])} author recording cases and fitted statistics to {destination}")
