"""Reference fixture for NaturalSpeech's modules from heatz123/naturalspeech (ca2f814), the community implementation that
fills the paper's gaps: its learnable upsampling (the Heatz123 variant, in eval mode with random BatchNorm statistics),
its duration predictor, its VAE memory bank and its soft-DTW over the pairwise KL matrix.

    docker run --rm -v <refsrc>:/ref -v <repo>:/repo -w /ref/naturalspeech gptsovits-ref \
        python /repo/tools/reference-data/naturalspeech_reference.py

Writes tests/AiDotNet.Tests/TextToSpeech/ReferenceData/naturalspeech_reference.json.
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.getcwd())

from models.losses import get_sdtw_kl_matrix  # noqa: E402
from models.models import DurationPredictor, LearnableUpsampling, VAEMemoryBank  # noqa: E402
from models.soft_dtw import compute_softdtw  # noqa: E402


def tolist(t):
    return t.detach().double().reshape(-1).tolist()


def main():
    torch.manual_seed(3)
    torch.set_default_dtype(torch.float64)
    out = {}

    # Learnable upsampling (proj_o is fixed at 192 → 384 in the port, so the width is 192).
    up = LearnableUpsampling(d_predictor=192, kernel_size=3, dropout=0.0, conv_output_size=8, dim_w=4, dim_c=2).double()
    with torch.no_grad():
        for m in up.modules():
            if isinstance(m, torch.nn.BatchNorm1d):
                m.running_mean.copy_(torch.randn_like(m.running_mean) * 0.2)
                m.running_var.copy_(torch.rand_like(m.running_var) + 0.5)
                m.weight.copy_(torch.rand_like(m.weight) + 0.5)
                m.bias.copy_(torch.randn_like(m.bias) * 0.1)
    up.eval()
    v = torch.randn(1, 5, 192)
    d = torch.tensor([[1.4, 2.2, 0.9, 3.1, 1.6]])
    with torch.no_grad():
        rep, mel_mask, mel_len, w = up(d, v, torch.tensor([5]), torch.zeros(1, 5, dtype=torch.bool), 5)
    out["upsampling"] = {"state": {k: tolist(x) for k, x in up.state_dict().items() if not k.endswith("num_batches_tracked")},
                         "hidden": tolist(v), "durations": d[0].tolist(), "frames": int(mel_len[0]), "output": tolist(rep[0])}

    dp = DurationPredictor(16, 12, 3, 0.5).double().eval()
    x = torch.randn(1, 16, 7)
    with torch.no_grad():
        logw = dp(x, torch.ones(1, 1, 7))
    out["duration"] = {"state": {k: tolist(t) for k, t in dp.state_dict().items()}, "hidden": tolist(x), "logw": tolist(logw)}

    mem = VAEMemoryBank(bank_size=10, n_hidden_dims=16, n_attn_heads=2).double().eval()
    z = torch.randn(1, 16, 6)
    with torch.no_grad():
        attended = mem(z)
    out["memory"] = {"state": {k: tolist(t) for k, t in mem.state_dict().items()}, "z": tolist(z), "output": tolist(attended)}

    # Soft-DTW (gamma 0.01, warp 0.07) over the pairwise KL matrix.
    zq = torch.randn(1, 4, 7)
    lq = torch.randn(1, 4, 7) * 0.2
    mp = torch.randn(1, 4, 5)
    lp = torch.randn(1, 4, 5) * 0.2
    kl = get_sdtw_kl_matrix(zq, lq, mp, lp)                                         # [b, t_p, t_q]
    r = compute_softdtw(kl.detach().numpy(), 0.01, 0.0, 0.07)
    out["softdtw"] = {"z": tolist(zq), "logs_q": tolist(lq), "m_p": tolist(mp), "logs_p": tolist(lp), "matrix": tolist(kl),
                      "value": float(r[0, kl.shape[1], kl.shape[2]])}
    path = os.path.join(HERE, "..", "..", "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", "naturalspeech_reference.json")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(out, handle)
        handle.write("\n")
    print("wrote", os.path.getsize(path), "bytes")


if __name__ == "__main__":
    main()
