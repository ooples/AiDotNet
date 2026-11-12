"""Reference fixture for ERes2NetV2's forward3 as GPT-SoVITS runs it (GPT_SoVITS/eres2net, sv.py), with Kaldi fbank.

Runs the release's vendored modules (RVC-Boss/GPT-SoVITS at b4570b8) in the gptsovits-ref image in float64: a small
ERes2NetV2 (m_channels 4, one block per stage, baseWidth 24, scale 4, expansion 4) in eval mode with random BatchNorm
statistics; records the state dict, a 0.6 s waveform, its fbank (eres2net/kaldi.py, 80 bins, no dither) and forward3.

    docker run --rm -v <refsrc>:/ref -v <repo>:/repo -w /ref/GPT-SoVITS/GPT_SoVITS/eres2net gptsovits-ref \
        python /repo/tools/reference-data/eres2netv2_reference.py

Writes tests/AiDotNet.Tests/Audio/ReferenceData/eres2netv2_reference.json.
"""
import json
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.getcwd())

from ERes2NetV2 import ERes2NetV2  # noqa: E402
import kaldi as Kaldi  # noqa: E402


def tolist(t):
    return t.detach().double().reshape(-1).tolist()


def main():
    torch.manual_seed(11)
    torch.set_default_dtype(torch.float64)
    model = ERes2NetV2(num_blocks=[1, 1, 1, 1], m_channels=4, baseWidth=24, scale=4, expansion=4).double()
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.running_mean.copy_(torch.randn_like(m.running_mean) * 0.1)
                m.running_var.copy_(torch.rand_like(m.running_var) + 0.5)
                m.weight.copy_(torch.rand_like(m.weight) + 0.5)
                m.bias.copy_(torch.randn_like(m.bias) * 0.1)
    model.eval()
    t = torch.arange(9600) / 16000.0
    wav = 0.3 * torch.sin(2 * math.pi * 220 * t) + 0.05 * torch.randn(9600)
    with torch.no_grad():
        feat = Kaldi.fbank(wav.unsqueeze(0), num_mel_bins=80, sample_frequency=16000, dither=0)
        embedding = model.forward3(feat.unsqueeze(0))
    # forward3 reads neither the pooling nor seg_1 (the 192-dim embedding head), so they are left out.
    out = {"state": {k: tolist(v) for k, v in model.state_dict().items()
                     if not k.endswith("num_batches_tracked") and not k.startswith(("seg_", "pool"))},
           "wav": tolist(wav), "fbank": tolist(feat), "fbank_shape": list(feat.shape), "embedding": tolist(embedding)}
    folder = os.path.join(HERE, "..", "..", "tests", "AiDotNet.Tests", "Audio", "ReferenceData")
    with open(os.path.join(folder, "eres2netv2_reference.json"), "w", encoding="utf-8", newline="\n") as handle:
        json.dump(out, handle)
        handle.write("\n")
    print("embedding", embedding.shape)


if __name__ == "__main__":
    main()
