"""Adafactor as transformers implements it (transformers.optimization.Adafactor, Shazeer & Stern 2018).

Records in tests/AiDotNet.Tests/UnitTests/Optimizers/ReferenceData/adafactor_reference.json, in float64, four steps of
three configurations on a [4, 6] parameter with fixed gradients: the paper's defaults (factored second moment, relative
step, parameter scaling, no first moment); PaLM's fine-tuning as AudioPaLM states it (a constant 5e-5 with parameter
scaling, beta1 0.9); and an unfactored second moment, which transformers keeps for a vector, so its parameter is the
same values flattened.

Usage (default: verify; --write: regenerate), in a venv with torch and transformers:
  python tools/reference-data/adafactor_reference.py [--write]
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fixture  # noqa: E402

FIXTURE = "tests/AiDotNet.Tests/UnitTests/Optimizers/ReferenceData/adafactor_reference.json"
CASES = {
    "relative": dict(lr=None, relative_step=True, scale_parameter=True, warmup_init=False, beta1=None, flat=False),
    "palm": dict(lr=5e-5, relative_step=False, scale_parameter=True, warmup_init=False, beta1=0.9, flat=False),
    "unfactored": dict(lr=None, relative_step=True, scale_parameter=True, warmup_init=False, beta1=0.9, flat=True),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    from transformers.optimization import Adafactor
    torch.set_default_dtype(torch.float64)
    stored = None if args.write else _fixture.load(FIXTURE)
    torch.manual_seed(3)
    initial = torch.tensor(stored["initial"]).reshape(4, 6) if stored else torch.randn(4, 6).float().double()
    grads = [torch.tensor(g).reshape(4, 6) for g in stored["gradients"]] if stored else \
        [(torch.randn(4, 6) * (0.5 + k)).float().double() for k in range(4)]
    out = {}
    for name, c in CASES.items():
        p = torch.nn.Parameter(initial.clone().reshape(-1) if c["flat"] else initial.clone())
        opt = Adafactor([p], lr=c["lr"], relative_step=c["relative_step"], scale_parameter=c["scale_parameter"],
                        warmup_init=c["warmup_init"], beta1=c["beta1"])
        steps = []
        for g in grads:
            p.grad = g.clone().reshape(p.shape)
            opt.step()
            steps.append([float(x) for x in p.detach().reshape(-1).tolist()])
        out[name] = steps
    if args.write:
        data = {"initial": _fixture.flat(initial), "gradients": [_fixture.flat(g) for g in grads]}
        data.update(out)
        _fixture.save(FIXTURE, data, __file__)
        print("wrote", FIXTURE)
        return
    check = _fixture.Comparison("Adafactor (transformers)")
    for name in CASES:
        for k, step in enumerate(out[name]):
            check.floats(f"{name}[{k}]", stored[name][k], step, dtype="float64")
    check.report()


if __name__ == "__main__":
    main()
