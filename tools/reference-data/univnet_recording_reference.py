"""Run the released UnivNet recording loader and mel transform without changing their bodies.

Requires torch, scipy and librosa. Git-blob hashes pin the three author files. AST selection
omits unrelated imports only. Compatibility adapters supply Torch's now-required return_complex
argument and librosa's keyword-only calling convention. Float32 WAV decoding and the author's
explicit float32 mel bank are promoted to float64 for the library's double-precision comparison.
Temporary WAV/mel evidence is retained; no model weights or fake feature backend are used.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
from urllib.request import urlopen

import librosa
import numpy as np
from scipy.io.wavfile import read, write
import torch
from torch.utils.data import Dataset

COMMIT = "9bb2b54838bb6d7ce767131cc7b8b61198bc7558"
FILES = {
    "datasets/dataloader.py": "37f17420eff07ce8449c6e837715c47ce3013cdb",
    "utils/stft.py": "ca588d0c4a568b433b300a7b362b4353c709b0cd",
    "utils/utils.py": "47677a57e79f21237279a20ec39fb00049f0a2fe",
}


def selected_source(path, names, namespace):
    url = f"https://raw.githubusercontent.com/maum-ai/univnet/{COMMIT}/{path}"
    with urlopen(url, timeout=60) as response:
        source = response.read()
    blob = hashlib.sha1(b"blob " + str(len(source)).encode() + b"\0" + source).hexdigest()
    if blob != FILES[path]:
        raise RuntimeError(f"Author source changed: {path}: {blob}")
    tree = ast.parse(source, filename=url)
    tree.body = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    if len(tree.body) != len(names):
        raise RuntimeError(f"Missing author definitions in {path}")
    exec(compile(ast.fix_missing_locations(tree), url, "exec"), namespace)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    original_stft = torch.stft

    def legacy_stft(*values, **kwargs):
        kwargs.setdefault("return_complex", False)
        return original_stft(*values, **kwargs)

    torch.stft = legacy_stft

    def legacy_mel(sr, n_fft, n_mels, fmin, fmax):
        return librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, dtype=np.float32)

    namespace = {"torch": torch, "np": np, "read": read}
    selected_source("utils/utils.py", {"read_wav_np"}, namespace)
    author_read = namespace["read_wav_np"]

    def double_input(path):
        rate, samples = author_read(path)
        return rate, samples.astype(np.float64)

    stft_namespace = {"torch": torch, "librosa_mel_fn": legacy_mel}
    selected_source("utils/stft.py", {"TacotronSTFT"}, stft_namespace)
    dataset_namespace = {"torch": torch, "np": np, "os": os, "random": random,
                         "Dataset": Dataset, "read_wav_np": double_input}
    selected_source("datasets/dataloader.py", {"MelFromDisk"}, dataset_namespace)
    evidence = Path(tempfile.mkdtemp(prefix="aidotnet-univnet-recording-reference-"))
    cases = []
    for name, length in [("short_zero_padded", 37), ("long_context_crop", 3072)]:
        sr, fft, hop, window, mels, segment, pad_short = 4000, 64, 16, 64, 8, 128, 2000
        samples = np.array([(((i * 37) % 211) - 105) / 256.0 + ((i % 7) - 3) / 64.0 for i in range(length)])
        wavpath = evidence / f"{name}.wav"
        write(wavpath, sr, (samples * 32768).astype(np.int16))
        stft = stft_namespace["TacotronSTFT"](fft, hop, window, mels, sr, 0.0, sr / 2).double()
        cls = dataset_namespace["MelFromDisk"]
        dataset = cls.__new__(cls)
        dataset.data_dir, dataset.meta, dataset.train = str(evidence), [(wavpath.name, "", "")], True
        dataset.mel_segment_length = segment // hop
        dataset.hp = SimpleNamespace(audio=SimpleNamespace(segment_length=segment, pad_short=pad_short,
            hop_length=hop, n_mel_channels=mels, sampling_rate=sr))
        dataset.stft = stft
        padded_length = max(length, segment + pad_short)
        maximum_start = padded_length // hop - segment // hop - 1
        seed = 3 if name.startswith("long") else next(s for s in range(4096) if random.Random(s).randint(0, maximum_start) == 0)
        start = random.Random(seed).randint(0, maximum_start)
        random.seed(seed)
        with torch.no_grad():
            mel, audio = dataset.my_getitem(0)
        decoded = double_input(str(wavpath))[1]
        padded = np.pad(decoded, (0, padded_length - length))
        assert np.array_equal(audio.numpy().reshape(-1), padded[start * hop:start * hop + segment])
        assert mel.shape == (mels, segment // hop)
        cases.append({"name": name, "raw_length": length, "padded_length": padded_length, "python_seed": seed,
            "start_frame": start, "maximum_start": maximum_start, "segment_samples": segment,
            "audio": audio.flatten().tolist(), "mel": mel.flatten().tolist()})
    data = {"commit": COMMIT, "git_blobs": FILES, "torch": torch.__version__, "librosa": librosa.__version__,
        "dtype": "float64, with the author's float32 mel basis promoted to float64", "cases": cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(cases)} unmodified-author loader cases; retained evidence in {evidence}")


if __name__ == "__main__":
    main()
