"""speechbrain's x-vector extractor (spkrec-xvect-voxceleb; speechbrain 0.5.16), the speaker embedding SpeechT5 reads.

Records in tests/AiDotNet.Tests/TextToSpeech/ReferenceData/speechbrain_xvector_reference.json, in float64, for a 0.4 s
waveform: the Fbank features after the sentence mean is removed, and the embedding of a tiny random Xvector (24 mels,
TDNN channels 8, 8, 8, 8, 12, embedding 6) with the pooling noise given; and, when --checkpoint is passed, the released
model's embedding (embedding_model.ckpt) of the same waveform with the same noise draws.

Usage (default: verify; --write: regenerate), in the speecht5-ref image:
  docker run --rm -v <repo>:/work -v <spkrec-xvect-voxceleb>:/ckpt:ro -w /work speecht5-ref \\
    python tools/reference-data/speechbrain_xvector_reference.py --checkpoint /ckpt/embedding_model.ckpt [--write]
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fixture  # noqa: E402

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/speechbrain_xvector_reference.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    import speechbrain.nnet.pooling as pooling
    from speechbrain.lobes.features import Fbank
    from speechbrain.processing.features import InputNormalization
    from speechbrain.lobes.models.Xvector import Xvector

    stored = None if args.write else _fixture.load(FIXTURE)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(512)
    wave = torch.tensor(stored["wave"]) if stored else (0.3 * torch.randn(6400) * torch.sin(torch.arange(6400) * 0.05)).float().double()
    noise = {}

    def gauss(shape):
        key = tuple(shape)
        g = torch.tensor(stored["noise"]).reshape(shape) if stored else torch.randn(shape).float().double()
        noise[key] = g
        g = g - torch.min(g)
        g = g / torch.max(g)
        return 1e-5 * ((1 - 9) * g + 9)
    pooling.StatisticsPooling._get_gauss_noise = lambda self, shape, device="cpu": gauss(shape)

    features = Fbank(n_mels=24)(wave.unsqueeze(0))
    features = InputNormalization(norm_type="sentence", std_norm=False)(features, torch.ones(1))
    tiny = Xvector(in_channels=24, tdnn_channels=[8, 8, 8, 8, 12], lin_neurons=6)
    with torch.no_grad():
        for name, p in tiny.named_parameters():
            p.copy_(torch.tensor(stored["state"][name]).reshape(p.shape) if stored else (0.3 * torch.randn_like(p)).float().double())
        for name, b in tiny.named_buffers():
            if "running_mean" in name:
                b.copy_(torch.tensor(stored["buffers"][name]) if stored else (0.1 * torch.randn_like(b)).float().double())
            elif "running_var" in name:
                b.copy_(torch.tensor(stored["buffers"][name]) if stored else (1 + 0.3 * torch.rand_like(b)).float().double())
    tiny.eval()
    with torch.no_grad():
        embedding = tiny(features, torch.ones(1))[0, 0]
    out = {"features": features[0], "embedding": embedding}
    if args.checkpoint:
        real = Xvector(in_channels=24)
        real.load_state_dict({k: v.double() if v.is_floating_point() else v for k, v in torch.load(args.checkpoint, map_location="cpu").items()})
        real.eval()
        with torch.no_grad():
            out["released_embedding"] = real(features, torch.ones(1))[0, 0]

    if args.write:
        data = {"wave": _fixture.flat(wave), "noise": _fixture.flat(noise[(1, 12)]),
                "released_noise": _fixture.flat(noise[(1, 1500)]) if (1, 1500) in noise else None,
                "state": {n: _fixture.flat(p) for n, p in tiny.named_parameters()},
                "buffers": {n: _fixture.flat(b) for n, b in tiny.named_buffers() if "running" in n}}
        data.update({k: [float(x) for x in v.reshape(-1).tolist()] for k, v in out.items()})
        _fixture.save(FIXTURE, data, __file__)
        print("wrote", FIXTURE, {k: list(v.shape) for k, v in out.items()})
        return
    check = _fixture.Comparison("speechbrain x-vector")
    for k, v in out.items():
        check.floats(k, stored[k], [float(x) for x in v.reshape(-1).tolist()], dtype="float64")
    check.report()


if __name__ == "__main__":
    main()
