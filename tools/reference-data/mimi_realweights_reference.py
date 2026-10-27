"""Real-weights reference for Mimi: the released codec (kyutai/moshiko-pytorch-bf16 at 2bfc9ae,
tokenizer-e351c8d8-checkpoint125.safetensors) loaded by moshi 0.2.2's get_mimi with 32 codebooks, as Sesame's CSM
loads it, run without streaming in float64 on three seconds of speech (MaskGCT's prompt.wav from Amphion, mixed to mono
and resampled to 24 kHz with librosa's soxr_hq). The rotary embedding and attention are replaced as in
mimi_reference.py.

Records the audio, its codes, the unquantized latent and the audio decoded from the codes.

Usage: python mimi_realweights_reference.py MIMI_SAFETENSORS (with ~/source/repos/.venv-csm and Amphion next to this
repository). Writes tests/AiDotNet.Tests/TextToSpeech/ReferenceData/mimi_realweights_reference.json.
"""
import json
import os
import sys

import librosa
import numpy as np
import soundfile
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mimi_reference  # noqa: E402,F401  (installs the float64 rotary embedding and attention, and the bitsandbytes stub)
from moshi.models import loaders  # noqa: E402

AMPHION = os.path.join(HERE, "..", "..", "..", "refsrc", "Amphion")


def main():
    model = loaders.get_mimi(sys.argv[1], device="cpu", num_codebooks=32).double().eval()
    for module in model.modules():
        if hasattr(module, "_embedding"):
            module._embedding = None
    wav, rate = soundfile.read(os.path.join(AMPHION, "models", "tts", "maskgct", "wav", "prompt.wav"), always_2d=True)
    mono = librosa.to_mono(wav.T.astype(np.float32))[: rate * 3]
    audio = librosa.resample(mono, orig_sr=rate, target_sr=24000, res_type="soxr_hq").astype(np.float64)
    with torch.no_grad():
        x = torch.from_numpy(audio).view(1, 1, -1)
        latent = model._encode_to_unquantized_latent(x)[0]
        codes = model.encode(x)[0]
        decoded = model.decode(codes[None])[0, 0]
    result = {
        "audio": audio.tolist(), "codes": codes.T.tolist(), "latent": latent.T.reshape(-1).tolist(),
        "latent_shape": [latent.shape[1], latent.shape[0]], "decoded": decoded.tolist(),
    }
    path = os.path.join(HERE, "..", "..", "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", "mimi_realweights_reference.json")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle)
        handle.write("\n")
    print(codes.shape, latent.shape, decoded.shape)


if __name__ == "__main__":
    main()
