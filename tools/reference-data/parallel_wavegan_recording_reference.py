"""Pinned, unmodified author preprocessing and Collater recording-training goldens."""
import ast
import hashlib
import json
import urllib.request
from pathlib import Path

import librosa
import numpy as np
import torch

COMMIT = "86740373ec609cb9fb192d472d2aea125041491a"
BLOBS = {
    "parallel_wavegan/bin/train.py": "5fab384adcf1f0461d918a021d33f9f2fc5f0295",
    "parallel_wavegan/bin/preprocess.py": "1654c3d1212d08f98c2defe0dc51f5c591126eb2",
}


def author(path, name, namespace):
    data = urllib.request.urlopen(
        f"https://raw.githubusercontent.com/kan-bayashi/ParallelWaveGAN/{COMMIT}/{path}"
    ).read()
    assert hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() == BLOBS[path]
    node = next(n for n in ast.parse(data).body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), namespace)
    return namespace[name]


preprocess = author("parallel_wavegan/bin/preprocess.py", "logmelfilterbank", dict(librosa=librosa, np=np))
Collater = author("parallel_wavegan/bin/train.py", "Collater", dict(torch=torch, np=np))
result = dict(commit=COMMIT, git_blobs=BLOBS, cases=[])
for segment, context, normalized in [(128, 2, False), (131, 2, True), (128, 0, False)]:
    length = 3077
    # Binary-exact samples also survive the author's explicit final float32 cast exactly.
    audio = np.array([((i*37)%211-105)/256.0+(i%7-3)/64.0 for i in range(length)], dtype=np.float64)
    mel = preprocess(audio, sampling_rate=4000, fft_size=64, hop_size=16,
        win_length=48, num_mels=8, fmin=0, fmax=2000)
    mean = np.array([m / 8.0 for m in range(8)])
    scale = np.array([1 + m / 4.0 for m in range(8)])
    if normalized:
        # The paper's explicitly fitted per-band standardization, applied before Collater.
        mel = (mel - mean) / scale
    collater = Collater(segment, 16, context, use_noise_input=True)
    np.random.seed(17); torch.manual_seed(17)
    start = int(np.random.randint(collater.start_offset, len(mel) + collater.end_offset))
    np.random.seed(17)
    inputs, target = collater([(audio, mel)])
    noise, condition = inputs
    assert noise.shape == target.shape
    result["cases"].append(dict(segment_samples=segment, context=context, normalized=normalized,
        start_frame=start, minimum_start=collater.start_offset, exclusive_end=len(mel)+collater.end_offset,
        raw_length=length, audio=target.flatten().tolist(), mel=condition.flatten().tolist(),
        means=mean.tolist() if normalized else None, scales=scale.tolist() if normalized else None))

short = np.zeros(128, dtype=np.float64)
features = preprocess(short, sampling_rate=4000, fft_size=64, hop_size=16,
    win_length=48, num_mels=8, fmin=0, fmax=2000)
assert len(features) <= Collater(128, 16, 2).mel_threshold
result["short_recording_removed_by_author_length_filter"] = True
result["rates_after_steps"] = {}
for group, rate in [("generator", 1e-4), ("discriminator", 5e-5)]:
    parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    optimizer = torch.optim.RAdam([parameter], lr=rate, eps=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    rates = []
    for step in range(6):
        parameter.grad = torch.ones_like(parameter)
        optimizer.step(); scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])
    result["rates_after_steps"][group] = rates
destination = Path(__file__).resolve().parents[2] / "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/parallel_wavegan_recording_reference.json"
destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {len(result['cases'])} author recording/context cases to {destination}")
