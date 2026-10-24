"""WeSpeaker ResNet reference: wenet-e2e/wespeaker (wespeaker/models/resnet.py, pooling_layers.py, and the front end of
wespeaker/cli/speaker.py), which StyleTTS-ZS's speaker-embedding feature matching (Li et al. 2024, §III-A) uses.

Builds a small randomly initialized ResNet (its BasicBlock at m = 4 channels, blocks {1, 2, 1, 1}; pass a checkout of
wenet-e2e/wespeaker with --wespeaker), stores its weights in float32 (batch-norm statistics moved off their defaults)
and records, in float64 and evaluation mode, for a 16 kHz waveform: the CLI's features (torchaudio.compliance.kaldi.fbank
of the int16-scaled waveform, 80 bins, Hamming window, mean-normalized), the stem's and every stage's feature map, and
the embedding. in tests/AiDotNet.Tests/TextToSpeech/ReferenceData/wespeaker_reference.json. Default: verify; --write.
"""
import argparse
import base64
import os
import sys
import types

import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

import _fixture

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/wespeaker_reference.json"
CHANNELS = 4
BLOCKS = [1, 2, 1, 1]
EMBEDDING = 16
SAMPLES = 8000


def load(root):
    # The package's __init__ imports the CLI (silero_vad, etc.); load the model modules without it.
    package = types.ModuleType("wespeaker")
    package.__path__ = [os.path.join(root, "wespeaker")]
    sys.modules["wespeaker"] = package
    models = types.ModuleType("wespeaker.models")
    models.__path__ = [os.path.join(root, "wespeaker", "models")]
    sys.modules["wespeaker.models"] = models
    import wespeaker.models.resnet as resnet
    return resnet


def build(resnet, state=None):
    torch.manual_seed(2294)
    model = resnet.ResNet(resnet.BasicBlock, BLOCKS, m_channels=CHANNELS, feat_dim=80, embed_dim=EMBEDDING)
    if state is None:
        g = torch.Generator().manual_seed(5)
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.running_mean.copy_(torch.randn(m.running_mean.shape, generator=g) * 0.1)
                    m.running_var.copy_(torch.rand(m.running_var.shape, generator=g) + 0.5)
                    m.weight.copy_(torch.rand(m.weight.shape, generator=g) + 0.5)
                    m.bias.copy_(torch.randn(m.bias.shape, generator=g) * 0.1)
    else:
        model.load_state_dict(state, strict=True)
    return model.eval()


def inputs():
    g = torch.Generator().manual_seed(7)
    t = torch.arange(SAMPLES, dtype=torch.float64) / 16000
    return (0.3 * torch.sin(2 * torch.pi * 210 * t) + 0.05 * torch.randn(SAMPLES, generator=g, dtype=torch.float64)).float()


def flat(t):
    return [float(v) for v in t.detach().reshape(-1).tolist()]


def run(model):
    import torchaudio.compliance.kaldi as kaldi
    wave = inputs().double()
    out = {}
    with torch.no_grad():
        feat = kaldi.fbank((wave * (1 << 15))[None], num_mel_bins=80, frame_length=25, frame_shift=10, sample_frequency=16000,
                           window_type="hamming")
        feat = feat - torch.mean(feat, dim=0)
        out["features"] = flat(feat)
        x = feat[None].permute(0, 2, 1).unsqueeze(1)
        x = torch.relu(model.bn1(model.conv1(x)))
        maps = [x]
        for layer in (model.layer1, model.layer2, model.layer3, model.layer4):
            x = layer(x)
            maps.append(x)
        out["maps"] = [flat(m) for m in maps]
        out["map_shapes"] = [list(m.shape) for m in maps]
        out["embedding"] = flat(model(feat[None])[1])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wespeaker", required=True, help="a checkout of wenet-e2e/wespeaker")
    parser.add_argument("--write", action="store_true", help="regenerate the fixture (default: verify it)")
    args = parser.parse_args()
    resnet = load(args.wespeaker)
    previous = torch.get_default_dtype()
    if args.write:
        state = {k: v.float().clone().contiguous() for k, v in build(resnet).state_dict().items()}
        torch.set_default_dtype(torch.float64)
        try:
            outputs = run(build(resnet, state).double())
        finally:
            torch.set_default_dtype(previous)
        data = {"channels": CHANNELS, "blocks": BLOCKS, "embedding_dim": EMBEDDING, "wave": flat(inputs()),
                "safetensors_base64": base64.b64encode(save_safetensors(state)).decode("ascii"), **outputs}
        _fixture.save(FIXTURE, data, __file__)
        print("wrote " + FIXTURE)
        return
    data = _fixture.load(FIXTURE)
    torch.set_default_dtype(torch.float64)
    try:
        again = run(build(resnet, load_safetensors(base64.b64decode(data["safetensors_base64"]))).double())
    finally:
        torch.set_default_dtype(previous)
    check = _fixture.Comparison("WeSpeaker ResNet")
    check.floats("features", data["features"], again["features"], 1e-12, float)
    for i, (a, b) in enumerate(zip(data["maps"], again["maps"])):
        check.floats(f"map {i}", a, b, 1e-12, float)
    check.floats("embedding", data["embedding"], again["embedding"], 1e-12, float)
    check.report(exact=False)


if __name__ == "__main__":
    main()
