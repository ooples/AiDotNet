"""FireRedTTS's streamable Mel codec (§3.3.2) against SoCodec's code (hhguo/SoCodec, socodec/networks/vqvae), on tiny
sizes, in float64.

The paper releases no code for its Mel codec; the port lays it out as SoCodec's StagedVQVAE (its Encoder, Decoder and
TreeVectorQuantization with four ordered groups), reading the mel instead of HuBERT features. This builds SoCodec's
Encoder(C=100, D=32, strides=[2]), its TreeVectorQuantization (8 codebooks of 8 codes, 4 groups, one per stream) and its
Decoder(C=100, D=32, strides=[2]), and records in tests/AiDotNet.Tests/TextToSpeech/ReferenceData/fireredtts_codec_reference.json
the encodings, the four streams' tokens and the decoded mel (the speaker embedding added) of a mel, and the decoding of
fixed tokens.

Usage: python fireredtts_codec_reference.py --socodec <SoCodec checkout> [--write]   (needs .venv-firered)
"""
import argparse
import os
import sys
import types

import torch

import _fixture

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/fireredtts_codec_reference.json"
MEL, D, FRAMES, K, DIM, GROUPS = 100, 32, 18, 8, 4, 4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socodec", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    root = os.path.abspath(args.socodec)
    # staged_vqvae's package imports pull in the vocoder and utilities; only the codec's modules are needed.
    for name, path in [("socodec", "socodec"), ("socodec.networks", "socodec/networks"), ("socodec.networks.common", "socodec/networks/common"),
                       ("socodec.networks.vqvae", "socodec/networks/vqvae"), ("socodec.utils", "socodec/utils")]:
        module = types.ModuleType(name)
        module.__path__ = [os.path.join(root, path)]
        sys.modules[name] = module
    sys.modules["socodec.networks.common.ecapa_tdnn"] = types.SimpleNamespace(ECAPA_TDNN=None)
    sys.modules["socodec.utils.audio"] = types.SimpleNamespace(TorchMelSpectrogram=None)
    import socodec.networks.vqvae.staged_vqvae as staged

    stored = None if args.write else _fixture.load(FIXTURE)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(3302)
    encoder = staged.Encoder(MEL, D, [2])
    decoder = staged.Decoder(MEL, D, [2])
    quantizer = staged.TreeVectorQuantization(D, vq_config={"codebook_size": [K, K], "codebook_dim": [DIM, DIM], "requires_projection": True,
                                                           "decay": 0.99, "threshold_ema_dead_code": 0},
                                              tree_config=[{"downsample_rate": 1, "n_groups": GROUPS, "dropout": 0, "ordered": True}])
    m = torch.nn.ModuleDict({"encoder": encoder, "decoder": decoder, "quantizer": quantizer}).eval()
    with torch.no_grad():
        for name, p in list(m.named_parameters()) + [(n, b) for n, b in m.named_buffers() if n.endswith(".embed")]:
            if stored is not None:
                p.copy_(torch.tensor(stored["state"][name], dtype=torch.float32).reshape(p.shape))
            elif name.endswith(".embed"):
                p.copy_((1.5 * torch.randn_like(p)).float())
            else:
                p.copy_((p * 3 + 0.02 * torch.randn_like(p)).float())

    def tensor(key, shape, std=1.0):
        if stored is not None:
            return torch.tensor(stored[key], dtype=torch.float32).reshape(shape).to(torch.float64)
        return (std * torch.randn(shape)).float().to(torch.float64)
    inputs = {"mel": tensor("mel", (FRAMES, MEL)), "speaker": tensor("speaker", (D,))}
    out = {}
    with torch.no_grad():
        encoded = encoder(inputs["mel"].unsqueeze(0))                                 # [1, F/2, D]
        out["encodings"] = encoded[0]
        quantized, (_, _, indices) = quantizer(encoded, torch.tensor([encoded.shape[1]]), update_codebook=False)
        tokens = indices[0][0]                                                         # [F/2, groups]
        out["quantized"] = quantized[0]
        mel, _ = decoder(quantized, inputs["speaker"].unsqueeze(0))
        out["decoded"] = mel[0]
        fixed = torch.tensor([[(5 * t + 11 * g) % (K * K) for g in range(GROUPS)] for t in range(FRAMES // 2)])
        out["fixed_quantized"] = quantizer.quantizers[0].decode(fixed.unsqueeze(0)).transpose(1, 2)[0]
    results = {"tokens": tokens.tolist(), "fixed_tokens": fixed.tolist()}
    if args.write:
        data = {"state": {n: _fixture.flat(p) for n, p in list(m.named_parameters()) + [(n, b) for n, b in m.named_buffers() if n.endswith(".embed")]},
                **results}
        data.update({k: _fixture.flat(v) for k, v in inputs.items()})
        data.update({k: [float(x) for x in v.reshape(-1).tolist()] for k, v in out.items()})
        _fixture.save(FIXTURE, data, __file__)
        print(f"wrote {FIXTURE}: tokens {results['tokens']}")
        return
    check = _fixture.Comparison("FireRedTTS Mel codec")
    for k, v in out.items():
        check.floats(k, stored[k], [float(x) for x in v.reshape(-1).tolist()], dtype="float64")
    for k, v in results.items():
        check.exact(k, stored[k], v)
    check.report()


if __name__ == "__main__":
    main()
