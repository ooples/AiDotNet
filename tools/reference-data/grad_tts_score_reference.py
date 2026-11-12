"""Export the author's complete Grad-TTS score U-Net, including input/parameter gradients.

Requires Python 3.12, torch==2.6.0 (CPU), einops==0.8.1, numpy==2.2.6.
Runs the unmodified, hash-checked author modules at the pinned commit below.
Weights use the same documented formulas as the .NET reference test; ReZero gates
are nonzero so the comparison exercises every linear-attention branch.
"""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import types
from urllib.request import urlopen

import torch
from torch import nn

COMMIT = "7782c7a223dd25df47b4538698ee9756774e2924"
BLOBS = {
    "base.py": "efc5d42d0d4775bf122f5ef397d4502c40679f64",
    "diffusion.py": "9717d3a38c824b1aef61511674fdcf48af138936",
}


def load_author():
    cache = Path(tempfile.gettempdir()) / ("aidotnet-grad-tts-author-" + COMMIT[:12])
    cache.mkdir(parents=True, exist_ok=True)
    for name, expected in BLOBS.items():
        path = cache / name
        if not path.exists():
            url = f"https://raw.githubusercontent.com/huawei-noah/Speech-Backbones/{COMMIT}/Grad-TTS/model/{name}"
            with urlopen(url, timeout=60) as response:
                path.write_bytes(response.read())
        data = path.read_bytes()
        actual = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Author source hash mismatch: {name}: {actual} != {expected}")
    package = types.ModuleType("model")
    package.__path__ = [str(cache)]
    sys.modules["model"] = package
    for name in ("base", "diffusion"):
        spec = importlib.util.spec_from_file_location("model." + name, cache / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return sys.modules["model.diffusion"]


def initialize(network):
    with torch.no_grad():
        for module in network.modules():
            if isinstance(module, nn.Linear):
                out_dim, in_dim = module.weight.shape
                row = torch.arange(out_dim, dtype=torch.float64)[:, None]
                col = torch.arange(in_dim, dtype=torch.float64)[None, :]
                module.weight.copy_(0.04 * torch.sin((row + 1) * (col + 1) * 0.13) / math.sqrt(in_dim))
            elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                a, b, height, width = module.weight.shape
                ia, ib, iy, ix = torch.meshgrid(
                    torch.arange(a, dtype=torch.float64), torch.arange(b, dtype=torch.float64),
                    torch.arange(height, dtype=torch.float64), torch.arange(width, dtype=torch.float64), indexing="ij")
                module.weight.copy_(0.03 * torch.sin((ia + 1) * (ib + 1) * 0.13 + iy * 0.17 + ix * 0.19)
                                    / math.sqrt(b * height * width))
            elif isinstance(module, nn.GroupNorm):
                index = torch.arange(module.weight.numel(), dtype=torch.float64) + 1
                module.weight.copy_(1 + 0.03 * torch.sin(index * 0.13))
                module.bias.copy_(0.01 * torch.cos(index * 0.17))
            elif type(module).__name__ == "Rezero":
                module.g.fill_(0.15)
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)) and module.bias is not None:
                index = torch.arange(module.bias.numel(), dtype=torch.float64) + 1
                module.bias.copy_(0.01 * torch.cos(index * 0.17))


def values(tensor):
    return tensor.detach().reshape(-1).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[2]
                        / "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/grad_tts_score_reference.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    author = load_author()
    network = author.GradLogPEstimator2d(8, dim_mults=(1, 2, 4), n_spks=1, n_feats=8, pe_scale=1000).double()
    initialize(network)
    frames, bins, valid = 12, 8, 10
    f = torch.arange(frames, dtype=torch.float64)[:, None]
    c = torch.arange(bins, dtype=torch.float64)[None, :]
    xt_rows = 0.3 * torch.sin(f * 0.31 + c * 0.17)
    mu_rows = 0.2 * torch.cos(f * 0.19 + c * 0.29)
    loss_weights = 0.4 + 0.03 * f + 0.01 * c
    xt = xt_rows.T[None].contiguous().requires_grad_()
    mu = mu_rows.T[None].contiguous().requires_grad_()
    mask = (torch.arange(frames) < valid).to(torch.float64)[None, None]
    output = network(xt, mask, mu, torch.tensor([0.37], dtype=torch.float64))
    loss = (output.square() * loss_weights.T[None]).sum()
    loss.backward()
    fixture = {
        "author_commit": COMMIT, "source_blobs": BLOBS, "torch_version": torch.__version__, "dtype": "float64",
        "dim": 8, "multipliers": [1, 2, 4], "position_scale": 1000, "time": 0.37,
        "frames": frames, "mel_bins": bins, "valid_frames": valid,
        "parameter_count": sum(p.numel() for p in network.parameters()),
        "xt": values(xt_rows), "mu": values(mu_rows), "loss_weights": values(loss_weights),
        "output": values(output[0].T.contiguous()), "loss": loss.item(),
        "xt_gradient": values(xt.grad[0].T.contiguous()), "mu_gradient": values(mu.grad[0].T.contiguous()),
        "first_conv_weight_gradient": values(network.downs[0][0].block1.block[0].weight.grad),
        "final_conv_weight_gradient": values(network.final_conv.weight.grad),
        "final_conv_bias_gradient": values(network.final_conv.bias.grad),
        # DenseLayer stores [input,output], opposite to torch.nn.Linear.
        "time_mlp_weight_gradient": values(network.mlp[0].weight.grad.T.contiguous()),
        "time_mlp_bias_gradient": values(network.mlp[0].bias.grad),
        "first_norm_weight_gradient": values(network.downs[0][0].block1.block[1].weight.grad),
        "first_norm_bias_gradient": values(network.downs[0][0].block1.block[1].bias.grad),
        # Match execution order: author registers the empty ups list before registering the middle blocks.
        "gate_names": [f"downs.{i}.2.fn.g" for i in range(len(network.downs))] + ["mid_attn.fn.g"]
                      + [f"ups.{i}.2.fn.g" for i in range(len(network.ups))],
        "gate_gradients": [block[2].fn.g.grad.item() for block in network.downs]
                          + [network.mid_attn.fn.g.grad.item()]
                          + [block[2].fn.g.grad.item() for block in network.ups],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}: {fixture['parameter_count']} author parameters, loss={loss.item():.17g}")


if __name__ == "__main__":
    main()
