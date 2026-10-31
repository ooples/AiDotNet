"""MinMo's input projector as FunASR 1.2.7 builds it (funasr/models/llm_asr/adaptor.py Transformer, with
funasr/models/transformer EncoderLayer, MultiHeadedAttention and PositionwiseFeedForward): a tiny adaptor (encoder width
6, downsample 2, bottleneck 10, LLM width 8, 2 layers of 2 heads) with seeded random weights, in float64, on 7 encoder
frames (odd, so the last chunk is zero-padded). Stores the state dict, the input and the output.
Usage: python tools/reference-data/minmo_projector_reference.py --funasr <dir holding funasr/models> [--write]
"""
import argparse
import os
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fixture  # noqa: E402

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/minmo_projector_reference.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--funasr")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    sys.path.insert(0, args.funasr)
    # The package's __init__ imports everything; register empty parents and load only the modules the adaptor needs.
    for name in ("funasr", "funasr.models", "funasr.register"):
        module = types.ModuleType(name)
        module.__path__ = [os.path.join(args.funasr, *name.split("."))]
        sys.modules[name] = module
    sys.modules["funasr.register"].tables = types.SimpleNamespace(register=lambda *a, **k: (lambda c: c))
    from funasr.models.llm_asr.adaptor import Transformer

    torch.manual_seed(0)
    adaptor = Transformer(downsample_rate=2, encoder_dim=6, llm_dim=8, ffn_dim=10, n_layer=2, attention_heads=2).double().eval()
    with torch.no_grad():
        for name, p in adaptor.named_parameters():
            if "norm" in name and name.endswith("weight"):
                p.copy_(1 + 0.2 * torch.randn_like(p))
            else:
                p.copy_(0.4 * torch.randn_like(p))
    x = torch.randn(1, 7, 6)
    with torch.no_grad():
        out, lens = adaptor(x, torch.tensor([7]))
    data = {"output": out.reshape(-1).tolist()}
    if args.write:
        state = {k: v.reshape(-1).tolist() for k, v in adaptor.state_dict().items()}
        _fixture.save(FIXTURE, {"input": x.reshape(-1).tolist(), "state": state, **data}, __file__)
        print("wrote", FIXTURE, out.shape, int(lens))
        return
    check = _fixture.Comparison("MinMo input projector")
    check.floats("output", _fixture.load(FIXTURE)["output"], data["output"], 1e-12, float)
    check.report(exact=False)


if __name__ == "__main__":
    main()
