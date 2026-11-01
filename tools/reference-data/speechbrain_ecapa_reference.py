"""speechbrain's ECAPA-TDNN extractor (spkrec-ecapa-voxceleb; speechbrain 0.5.16), the speaker embedding WhisperSpeech reads.

Records in tests/AiDotNet.Tests/TextToSpeech/ReferenceData/speechbrain_ecapa_reference.json, in float64, for a 0.5 s
waveform: the Fbank features after the sentence mean is removed, and the embedding (encode_batch: features, sentence mean
normalization, ECAPA_TDNN with every frame valid) of a tiny random ECAPA_TDNN (80 mels, channels 12, 16, 16, 16, 48 so the
first SE-Res2Net block has a shortcut, kernels 5, 3, 3, 3, 1, dilations 1, 2, 3, 4, 1, Res2Net scale 8, SE and attention
channels 8, embedding 6) with non-trivial batch-norm statistics; and, when --checkpoint is passed, the released model's
embedding (embedding_model.ckpt) of the same waveform.

Usage (default: verify; --write: regenerate), in the speecht5-ref image:
  docker run --rm -v <repo>:/work -v <spkrec-ecapa-voxceleb>:/ckpt:ro -w /work speecht5-ref \\
    python tools/reference-data/speechbrain_ecapa_reference.py [--checkpoint /ckpt/embedding_model.ckpt] [--write]
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fixture  # noqa: E402

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/speechbrain_ecapa_reference.json"
TINY = dict(input_size=80, channels=[12, 16, 16, 16, 48], kernel_sizes=[5, 3, 3, 3, 1], dilations=[1, 2, 3, 4, 1], attention_channels=8,
            res2net_scale=8, se_channels=8, lin_neurons=6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    from speechbrain.lobes.features import Fbank
    from speechbrain.processing.features import InputNormalization
    from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN

    stored = None if args.write else _fixture.load(FIXTURE)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(513)
    wave = torch.tensor(stored["wave"]) if stored else (0.3 * torch.randn(8000) * torch.sin(torch.arange(8000) * 0.03)).float().double()

    features = Fbank(n_mels=80)(wave.unsqueeze(0))
    features = InputNormalization(norm_type="sentence", std_norm=False)(features, torch.ones(1))
    tiny = ECAPA_TDNN(**TINY)
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
        real = ECAPA_TDNN(input_size=80, channels=[1024, 1024, 1024, 1024, 3072], kernel_sizes=[5, 3, 3, 3, 1], dilations=[1, 2, 3, 4, 1],
                          attention_channels=128, lin_neurons=192)
        real.load_state_dict({k: v.double() if v.is_floating_point() else v for k, v in torch.load(args.checkpoint, map_location="cpu").items()})
        real.eval()
        with torch.no_grad():
            out["released_embedding"] = real(features, torch.ones(1))[0, 0]

    if args.write:
        data = {"config": TINY, "wave": _fixture.flat(wave),
                "state": {n: _fixture.flat(p) for n, p in tiny.named_parameters()},
                "buffers": {n: _fixture.flat(b) for n, b in tiny.named_buffers() if "running" in n}}
        data.update({k: [float(x) for x in v.reshape(-1).tolist()] for k, v in out.items()})
        _fixture.save(FIXTURE, data, __file__)
        print("wrote", FIXTURE, {k: list(v.shape) for k, v in out.items()}, embedding.tolist())
        return
    check = _fixture.Comparison("speechbrain ECAPA")
    for key, value in out.items():
        if key in stored:
            check.floats(key, stored[key], value.reshape(-1).tolist(), 1e-12, float)
    check.report(exact=False)


if __name__ == "__main__":
    main()
