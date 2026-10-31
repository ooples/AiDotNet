"""D-Adaptation Adam reference: audiocraft's DAdaptAdam (audiocraft/optim/dadam.py, from facebookresearch/dadaptation).

Runs the reference optimizer on fixed gradient sequences (seeded, independent of the parameters so the C# test can
replay them) for two configurations: MusicGen's (lr 1, betas (0.9, 0.95), decoupled weight decay 0.1) and one with
coupled weight decay and a learning-rate multiplier. Records the parameters and d after every step in
tests/AiDotNet.Tests/UnitTests/Optimizers/ReferenceData/dadapt_adam_reference.json.
Pass an audiocraft checkout with --audiocraft. Default: verify; --write: regenerate.
"""
import argparse
import json
import os
import sys
import types

FIXTURE = "tests/AiDotNet.Tests/UnitTests/Optimizers/ReferenceData/dadapt_adam_reference.json"

CONFIGS = [
    {"name": "musicgen", "lr": 1.0, "betas": [0.9, 0.95], "eps": 1e-8, "weight_decay": 0.1, "decouple": True,
     "d0": 1e-6, "steps": 12, "size": 7, "seed": 11},
    {"name": "coupled", "lr": 0.5, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01, "decouple": False,
     "d0": 1e-4, "steps": 10, "size": 5, "seed": 23},
]


def compute(audiocraft):
    # audiocraft imports xformers at package import; the optimizer never uses it, so a stub module suffices.
    stub = types.ModuleType("xformers")
    stub.ops = types.ModuleType("xformers.ops")
    sys.modules.setdefault("xformers", stub)
    sys.modules.setdefault("xformers.ops", stub.ops)
    sys.path.insert(0, audiocraft)
    from audiocraft.optim.dadam import DAdaptAdam
    import torch
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    cases = []
    for c in CONFIGS:
        g = torch.Generator().manual_seed(c["seed"])
        start = torch.randn(c["size"], generator=g)
        gradients = [torch.randn(c["size"], generator=g) for _ in range(c["steps"])]
        p = torch.nn.Parameter(start.clone())
        opt = DAdaptAdam([p], lr=c["lr"], betas=tuple(c["betas"]), eps=c["eps"], weight_decay=c["weight_decay"],
                         decouple=c["decouple"], d0=c["d0"])
        trajectory = []
        for grad in gradients:
            p.grad = grad.clone()
            opt.step()
            trajectory.append({"params": p.detach().tolist(), "d": opt.param_groups[0]["d"]})
        cases.append({**c, "start": start.tolist(), "gradients": [x.tolist() for x in gradients], "trajectory": trajectory})
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--audiocraft", default=os.path.expanduser("~/source/repos/ref/audiocraft"))
    args = parser.parse_args()
    cases = compute(args.audiocraft)
    if args.write:
        os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
        with open(FIXTURE, "w", encoding="utf-8") as f:
            json.dump({"reference": "audiocraft/optim/dadam.py DAdaptAdam, float64", "cases": cases}, f)
        print(f"wrote {len(cases)} cases")
        return
    with open(FIXTURE, encoding="utf-8") as f:
        stored = json.load(f)["cases"]
    worst = 0.0
    for c, s in zip(cases, stored):
        for a, b in zip(c["trajectory"], s["trajectory"]):
            worst = max(worst, max(abs(x - y) for x, y in zip(a["params"], b["params"])), abs(a["d"] - b["d"]))
    print(f"max difference {worst:.3g}")
    sys.exit(0 if worst < 1e-12 else 1)


if __name__ == "__main__":
    main()
