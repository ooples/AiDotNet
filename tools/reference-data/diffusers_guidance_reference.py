"""Reference outputs for AdaptiveProjectedGuidance and RescaledCFG (diffusers 0.35).

APG is diffusers' normalized_guidance with eta = 1 - projectionStrength (0.5) and scale 7.5; RescaledCFG is
rescale_noise_cfg (guidance_rescale 0.7) applied to standard CFG at scale 7.5. Both are per sample. Used by
tests/AiDotNet.Tests/UnitTests/Diffusion/Guidance/DiffusersGuidanceReferenceTests.cs.
"""
import os
import re

import diffusers
import torch
from diffusers.guiders.adaptive_projected_guidance import normalized_guidance

_src = open(os.path.join(os.path.dirname(diffusers.__file__), "pipelines", "stable_diffusion",
                         "pipeline_stable_diffusion.py"), encoding="utf-8").read()
exec(re.search(r"def rescale_noise_cfg\(.*?\n    return noise_cfg\n", _src, re.S).group(0))

torch.set_default_dtype(torch.float64)
u = torch.tensor([[0.1, -0.4, 0.7, 0.2, -0.3, 0.5], [1.2, -0.8, 0.3, 0.0, 0.6, -1.1]]).reshape(2, 1, 2, 3)
c = torch.tensor([[0.3, -0.1, 0.9, -0.2, 0.1, 0.4], [0.7, -0.2, 0.8, 0.5, 0.1, -0.9]]).reshape(2, 1, 2, 3)


def fmt(x):
    return "{ " + ", ".join(f"{v:.15g}" for v in x.flatten().tolist()) + " }"


print("APG", fmt(normalized_guidance(c, u, 7.5, eta=1 - 0.5)))
g = u + 7.5 * (c - u)
print("RESCALE", fmt(rescale_noise_cfg(g, c, guidance_rescale=0.7)))
