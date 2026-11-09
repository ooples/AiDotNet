"""Chatterbox's HiFT sine source reference (resemble-ai/chatterbox 5de7a54, models/s3gen/hifigan.py): S3Gen keeps the
original SineGen at 24 kHz, where CosyVoice 2 switches to SineGen2. Records the harmonics of an F0 contour (voiced and
unvoiced frames, 480 samples a frame) with the starting phases (Uniform(-pi, pi).sample) and noise (torch.randn_like)
given, in float64 (the voicing kept in F0's dtype).
Usage: python tools/reference-data/chatterbox_sinegen_reference.py --chatterbox <checkout>/src [--write]
"""
import argparse
import os
import sys
from unittest import mock

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fixture  # noqa: E402

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/chatterbox_sinegen_reference.json"
F0 = [0.0, 120.0, 135.5, 150.25, 0.0, 0.0, 210.0, 205.0, 8.0, 180.0]
SCALE = 480
HARMONICS = 3


def sinegen(phases, noise):
    from torch.distributions.uniform import Uniform
    from chatterbox.models.s3gen.hifigan import SineGen
    gen = SineGen(24000, HARMONICS - 1, 0.1, 0.003, 10)
    gen._f02uv = lambda f0: (f0 > gen.voiced_threshold).to(f0.dtype)
    f0 = torch.tensor(F0, dtype=torch.float64).repeat_interleave(SCALE).reshape(1, 1, -1)
    with mock.patch.object(Uniform, "sample", lambda self, sample_shape=(): phases.clone().reshape(1, HARMONICS, 1)), \
            mock.patch.object(torch, "randn_like", return_value=noise.clone()):
        sines, _, _ = gen(f0)
    return sines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chatterbox", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, args.chatterbox)
    torch.set_default_dtype(torch.float64)
    if args.write:
        g = torch.Generator().manual_seed(11)
        phases = torch.rand(1, HARMONICS, generator=g) * 2 * torch.pi - torch.pi
        noise = torch.randn(1, HARMONICS, len(F0) * SCALE, generator=g)
    else:
        stored = _fixture.load(FIXTURE)
        phases = torch.tensor(stored["phases"]).reshape(1, HARMONICS)
        noise = torch.tensor(stored["noise"]).reshape(1, HARMONICS, len(F0) * SCALE)
    sines = sinegen(phases, noise).reshape(-1).tolist()
    if args.write:
        _fixture.save(FIXTURE, {"f0": F0, "scale": SCALE, "harmonics": HARMONICS, "phases": phases.reshape(-1).tolist(),
                                "noise": noise.reshape(-1).tolist(), "sines": sines}, __file__)
        print("wrote", FIXTURE)
        return
    check = _fixture.Comparison("Chatterbox SineGen")
    check.floats("sines", stored["sines"], sines, 1e-12, float)
    check.report(exact=False)


if __name__ == "__main__":
    main()
