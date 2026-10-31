"""SenseVoice-Small's encoder reference (An et al. 2024; the release's SenseVoiceEncoderSmall from QwenAudio/SenseVoice
model.py, and FunASR 1.2.7's WavFrontend: kaldi fbank on 16-bit-scaled samples, apply_lfr, apply_cmvn).

A tiny encoder (8 mel bands, LFR 7/6, width 16, 2 heads, feed-forward 32, 3 blocks then 2 tp blocks, FSMN kernel 11) with
seeded random weights and normalization, in float64, on 1.3 s of a chirp. Dither is 0. Stores the state dict, the
normalization, the features and the encoder output.
Usage: python tools/reference-data/sensevoice_encoder_reference.py --sensevoice <SenseVoice checkout> --funasr <dir holding
funasr/frontends/wav_frontend.py> [--write]
"""
import argparse
import math
import os
import sys
import types

import torch
import torchaudio.compliance.kaldi as kaldi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fixture  # noqa: E402

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/sensevoice_encoder_reference.json"
MELS, LFR_M, LFR_N = 8, 7, 6
CONF = dict(output_size=16, attention_heads=2, linear_units=32, num_blocks=3, tp_blocks=2, kernel_size=11, dropout_rate=0.0,
            attention_dropout_rate=0.0)


def load_module(path, name, stubs):
    for stub in stubs:
        sys.modules.setdefault(stub, types.ModuleType(stub))
    # apply_lfr and apply_cmvn cast to float32; the comparison keeps float64.
    source = open(path, encoding="utf-8").read().replace(".clone().type(torch.float32)", ".clone()")
    module = types.ModuleType(name)
    exec(compile(source, path, "exec"), module.__dict__)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensevoice")
    parser.add_argument("--funasr")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    # Only the frontend helpers and the encoder classes are used; their package imports are stubbed.
    register = types.ModuleType("funasr.register")
    register.tables = types.SimpleNamespace(register=lambda *a, **k: (lambda c: c))
    sys.modules["funasr"] = types.ModuleType("funasr")
    sys.modules["funasr.register"] = register
    sys.modules["funasr.frontends"] = types.ModuleType("funasr.frontends")
    sys.modules["funasr.frontends.eend_ola_feature"] = types.ModuleType("funasr.frontends.eend_ola_feature")
    frontend = load_module(os.path.join(args.funasr, "funasr", "frontends", "wav_frontend.py"), "wav_frontend", [])
    source = open(os.path.join(args.sensevoice, "model.py"), encoding="utf-8").read()
    encoder_module = types.ModuleType("sensevoice_encoder")
    encoder_module.tables = register.tables
    # The release's LayerNorm computes in float32; the comparison keeps float64.
    source = source.replace("input.float()", "input").replace("self.weight.float()", "self.weight").replace("self.bias.float()", "self.bias")
    start = source.index("class SinusoidalPositionEncoder")
    end = source.index("@tables.register(\"model_classes\", \"SenseVoiceSmall\")")
    exec("import torch\nfrom torch import nn\nimport torch.nn.functional as F\nfrom typing import Optional\n" + source[start:end],
         encoder_module.__dict__)

    torch.manual_seed(0)
    encoder = encoder_module.SenseVoiceEncoderSmall(input_size=MELS * LFR_M, **CONF).double().eval()
    with torch.no_grad():
        for name, p in encoder.named_parameters():
            if name.endswith("norm1.weight") or name.endswith("norm2.weight") or name.endswith("norm.weight"):
                p.copy_(1 + 0.2 * torch.randn_like(p))
            else:
                p.copy_(0.3 * torch.randn_like(p))
    shift = (-8 + 0.5 * torch.randn(MELS * LFR_M)).tolist()
    scale = (0.3 + 0.05 * torch.rand(MELS * LFR_M)).tolist()

    n = 20800
    i = torch.arange(n, dtype=torch.float64)
    audio = 0.4 * torch.sin(2 * math.pi * 150 * i / 16000 * (1 + i / 30000)) + 0.1 * torch.sin(0.7 * i)
    mat = kaldi.fbank((audio * (1 << 15)).unsqueeze(0), num_mel_bins=MELS, frame_length=25, frame_shift=10, dither=0.0,
                      energy_floor=0.0, window_type="hamming", sample_frequency=16000, snip_edges=True)
    mat = frontend.apply_lfr(mat, LFR_M, LFR_N).double()
    cmvn = torch.tensor([shift, scale], dtype=torch.float64)
    feats = mat + cmvn[0:1]
    feats = feats * cmvn[1:2]
    with torch.no_grad():
        out, _ = encoder(feats.unsqueeze(0).clone(), torch.tensor([feats.shape[0]]))
    data = {"features": feats.reshape(-1).tolist(), "frames": feats.shape[0], "output": out.reshape(-1).tolist()}
    if args.write:
        state = {k: v.reshape(-1).tolist() for k, v in encoder.state_dict().items()}
        _fixture.save(FIXTURE, {"audio": audio.tolist(), "shift": shift, "scale": scale, "state": state, **data}, __file__)
        print("wrote", FIXTURE, feats.shape, out.shape)
        return
    stored = _fixture.load(FIXTURE)
    check = _fixture.Comparison("SenseVoice encoder")
    check.floats("features", stored["features"], data["features"], 1e-12, float)
    check.floats("output", stored["output"], data["output"], 1e-12, float)
    check.report(exact=False)


if __name__ == "__main__":
    main()
