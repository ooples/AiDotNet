"""Reference fixture for SNAC (Siuzdak et al. 2024), the codec Orpheus speaks through: snac.SNAC with the released
hubertsiuzdak/snac_24khz configuration, in float64.

Every parameter is set by a closed form of its name and flat index i, so the C# test rebuilds the full-size codec without
a stored checkpoint: a Snake alpha is 1 + 0.5 * sin(0.37 * i + 0.11 * len(name)), a weight-norm gain (original0) is
0.5 + 0.2 * sin(...), and everything else (directions, biases, codebooks) is 0.05 * sin(...). The NoiseBlocks draw from a
seeded generator and the draws are recorded in order, so the C# decoder replays them.

Records the codes of a 5000-sample signal (padded to whole 2048-sample windows as SNAC.preprocess does), the audio
SNAC.decode makes of them, and each parameter's name (the checkpoint's own, parametrizations.weight.original0/1) and size.

Needs a clone of hubertsiuzdak/snac at 8f79a71 next to this repository (../refsrc/snac). Run with torch and einops.
Writes tests/AiDotNet.Tests/TextToSpeech/ReferenceData/snac_reference.json.
"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "refsrc", "snac"))
torch.set_default_dtype(torch.float64)

from snac import SNAC  # noqa: E402

CONFIG = {
    "sampling_rate": 24000, "encoder_dim": 48, "encoder_rates": [2, 4, 8, 8], "decoder_dim": 1024,
    "decoder_rates": [8, 8, 4, 2], "attn_window_size": None, "codebook_size": 4096, "codebook_dim": 8,
    "vq_strides": [4, 2, 1], "noise": True, "depthwise": True,
}


def closed_form(name, count):
    i = torch.arange(count, dtype=torch.float64)
    s = torch.sin(0.37 * i + 0.11 * len(name))
    if name.endswith(".alpha"):
        return 1 + 0.5 * s
    if name.endswith("original0"):
        return 0.5 + 0.2 * s
    return 0.05 * s


def main():
    model = SNAC(**CONFIG).eval()
    names = {name: parameter.numel() for name, parameter in model.named_parameters()}
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(closed_form(name, parameter.numel()).reshape(parameter.shape))
    n = 5000
    t = torch.arange(n, dtype=torch.float64)
    audio = (0.4 * torch.sin(0.031 * t) + 0.2 * torch.sin(0.17 * t + 0.3) + 0.05 * torch.cos(0.9 * t)).reshape(1, 1, n)

    generator = torch.Generator().manual_seed(7)
    noise = []
    real_randn = torch.randn

    def recording_randn(*size, **kwargs):
        kwargs.pop("device", None)
        kwargs["generator"] = generator
        value = real_randn(*size, **kwargs)
        noise.append(value.flatten().tolist())
        return value

    with torch.no_grad():
        codes = model.encode(audio)
        torch.randn = recording_randn
        try:
            decoded = model.decode(codes)
        finally:
            torch.randn = real_randn
    result = {
        "config": CONFIG,
        "audio": audio.flatten().tolist(),
        "codes": [c.flatten().tolist() for c in codes],
        "noise": noise,
        "decoded": decoded.flatten().tolist(),
        "names": names,
    }
    path = os.path.join(HERE, "..", "..", "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", "snac_reference.json")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle)
        handle.write("\n")
    print("codes:", [len(c.flatten()) for c in codes], "decoded:", decoded.shape[-1], "noise draws:", len(noise),
          "distinct level-0 codes:", len(set(codes[0].flatten().tolist())))


if __name__ == "__main__":
    main()
