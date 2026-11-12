"""Reference fixture for ScaledAdam: GPT-SoVITS's AR/modules/optim.py (icefall's) stepping fixed parameters with fixed
gradients in float64.

Parameters: a [3, 4] matrix, a [5] vector, two scalars of shape [1] (stacked by the reference into one batch) and a
lone [2, 1] tensor; 30 steps with clipping (scale 2, update period 5, so the threshold engages), size update period 4,
lr 0.01, betas (0.9, 0.95). Records every parameter after each step.

    docker run --rm -v <refsrc>:/ref -v <repo>:/repo -w /ref/GPT-SoVITS/GPT_SoVITS gptsovits-ref \
        python /repo/tools/reference-data/scaled_adam_reference.py

Writes tests/AiDotNet.Tests/UnitTests/Optimizers/ReferenceData/scaled_adam_reference.json.
"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.getcwd())

from AR.modules.optim import ScaledAdam  # noqa: E402


def main():
    torch.manual_seed(7)
    torch.set_default_dtype(torch.float64)
    names = ["matrix", "vector", "scalar_a", "scalar_b", "column"]
    params = [torch.nn.Parameter(torch.randn(3, 4)), torch.nn.Parameter(torch.randn(5) * 0.5),
              torch.nn.Parameter(torch.tensor([0.7])), torch.nn.Parameter(torch.tensor([-1.3])),
              torch.nn.Parameter(torch.randn(2, 1))]
    initial = {n: p.detach().reshape(-1).tolist() for n, p in zip(names, params)}
    optimizer = ScaledAdam(params, lr=0.01, betas=(0.9, 0.95), clipping_scale=2.0, parameters_names=[names],
                           show_dominant_parameters=False, clipping_update_period=5)
    steps = []
    gradients = []
    for step in range(30):
        grads = [torch.randn_like(p) * (1.0 + 3.0 * (step % 7 == 0)) for p in params]
        for p, g in zip(params, grads):
            p.grad = g.clone()
        optimizer.step()
        gradients.append({n: g.reshape(-1).tolist() for n, g in zip(names, grads)})
        steps.append({n: p.detach().reshape(-1).tolist() for n, p in zip(names, params)})
    out = {"names": names, "shapes": {n: list(p.shape) for n, p in zip(names, params)}, "initial": initial,
           "gradients": gradients, "steps": steps}
    folder = os.path.join(HERE, "..", "..", "tests", "AiDotNet.Tests", "UnitTests", "Optimizers", "ReferenceData")
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "scaled_adam_reference.json"), "w", encoding="utf-8", newline="\n") as handle:
        json.dump(out, handle)
        handle.write("\n")
    print("wrote", len(steps), "steps")


if __name__ == "__main__":
    main()
