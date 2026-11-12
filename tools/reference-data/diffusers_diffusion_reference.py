"""Reference outputs from diffusers 0.35 for the diffusion scheduler and guidance tests.

Schedulers (Euler and DPM-Solver++ multistep):

Runs each scheduler for 10 trailing-spaced steps (timesteps 999, 899, ..., 99, which is the spacing
NoiseSchedulerBase.SetTimesteps uses) over Stable Diffusion's scaled-linear betas in float64, with a fixed toy
model output m(x, t) = 0.5 * x + c * (1 + t / 1000) (epsilon or v), and prints the final samples used by
tests/AiDotNet.Tests/UnitTests/Diffusion/Schedulers/DiffusersSchedulerReferenceTests.cs.

Guidance: APG is diffusers' normalized_guidance with eta = 1 - projectionStrength (0.5) and scale 7.5;
RescaledCFG is rescale_noise_cfg (guidance_rescale 0.7) applied to standard CFG at scale 7.5. Both are per
sample. Used by tests/AiDotNet.Tests/UnitTests/Diffusion/Guidance/DiffusersGuidanceReferenceTests.cs.

Consistency models: one CMStochasticIterativeScheduler step (Song et al. 2023, Algorithm 1) at the first of two
Karras sigmas (sigma_min 0.002, sigma_max 80, rho 7) re-noises the boundary-conditioned denoised sample to
sigma_next with z = sin(1.3 (i + 1)). Prints sigma_next, the denoised sample and the re-noised sample used by
DiffusersSchedulerReferenceTests.ConsistencyRenoise_MatchesDiffusers.
"""
import os
import re

import diffusers
import torch
import diffusers.schedulers.scheduling_euler_ancestral_discrete as euler_ancestral
import diffusers.schedulers.scheduling_consistency_models as consistency_models
from diffusers import CMStochasticIterativeScheduler, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler
from diffusers.guiders.adaptive_projected_guidance import normalized_guidance

torch.set_default_dtype(torch.float64)
X0 = torch.tensor([1.0, -0.5, 0.25, 2.0])
C = torch.tensor([0.1, -0.2, 0.3, -0.4])


def to_float64(s):
    betas = torch.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000, dtype=torch.float64) ** 2
    s.betas = betas
    s.alphas = 1.0 - betas
    s.alphas_cumprod = torch.cumprod(s.alphas, dim=0)
    if hasattr(s, "alpha_t"):
        s.alpha_t = torch.sqrt(s.alphas_cumprod)
        s.sigma_t = torch.sqrt(1 - s.alphas_cumprod)
        s.lambda_t = torch.log(s.alpha_t) - torch.log(s.sigma_t)
    return s


_tensor_to = torch.Tensor.to


def _to_keep_float64(self, *args, **kwargs):
    # diffusers upcasts step math to float32 ("to avoid precision issues"); keep float64 so the reference is exact.
    args = tuple(torch.float64 if a is torch.float32 else a for a in args)
    if kwargs.get("dtype") is torch.float32:
        kwargs["dtype"] = torch.float64
    return _tensor_to(self, *args, **kwargs)


torch.Tensor.to = _to_keep_float64


_draws = []


def _noise(shape, **kwargs):
    # Euler ancestral noise for step k is sin(1.3 (k + 1) (i + 1)), so the C# test can pass the same vectors.
    k = len(_draws)
    _draws.append(k)
    return torch.sin(1.3 * (k + 1) * torch.arange(1, shape[-1] + 1, dtype=torch.float64))


euler_ancestral.randn_tensor = _noise


def run(s):
    _draws.clear()
    s.set_timesteps(10)
    # diffusers stores the inference sigmas as float32; rebuild them in float64 exactly as set_timesteps does
    # (an interpolation at integer timesteps is a gather, then a trailing zero for final_sigmas_type="zero").
    ac = s.alphas_cumprod
    s.sigmas = torch.cat([((1 - ac) / ac)[s.timesteps.long()] ** 0.5, torch.zeros(1)])
    x = X0.clone()
    for t in s.timesteps:
        eps = 0.5 * x + C * (1.0 + int(t) / 1000.0)
        x = s.step(eps, t, x).prev_sample
    return [int(t) for t in s.timesteps], x


common = dict(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
              timestep_spacing="trailing")
for name, sched in [
    ("Euler", EulerDiscreteScheduler(**common)),
    ("EulerV", EulerDiscreteScheduler(prediction_type="v_prediction", **common)),
    ("EulerAncestral", EulerAncestralDiscreteScheduler(**common)),
    ("EulerAncestralV", EulerAncestralDiscreteScheduler(prediction_type="v_prediction", **common)),
    ("DPMSolverPP2", DPMSolverMultistepScheduler(solver_order=2, **common)),
    ("DPMSolverPP3", DPMSolverMultistepScheduler(solver_order=3, **common)),
]:
    ts, x = run(to_float64(sched))
    print(name, ts, "{ " + ", ".join(f"{v:.15g}" for v in x.tolist()) + " }")

# Guidance (batch of two samples)
_src = open(os.path.join(os.path.dirname(diffusers.__file__), "pipelines", "stable_diffusion",
                         "pipeline_stable_diffusion.py"), encoding="utf-8").read()
exec(re.search(r"def rescale_noise_cfg\(.*?\n    return noise_cfg\n", _src, re.S).group(0))

u = torch.tensor([[0.1, -0.4, 0.7, 0.2, -0.3, 0.5], [1.2, -0.8, 0.3, 0.0, 0.6, -1.1]]).reshape(2, 1, 2, 3)
c = torch.tensor([[0.3, -0.1, 0.9, -0.2, 0.1, 0.4], [0.7, -0.2, 0.8, 0.5, 0.1, -0.9]]).reshape(2, 1, 2, 3)


def fmt(x):
    return "{ " + ", ".join(f"{v:.15g}" for v in x.flatten().tolist()) + " }"


print("APG", fmt(normalized_guidance(c, u, 7.5, eta=1 - 0.5)))
g = u + 7.5 * (c - u)
print("RESCALE", fmt(rescale_noise_cfg(g, c, guidance_rescale=0.7)))

# Consistency model re-noise (one multistep step)
consistency_models.randn_tensor = lambda shape, **kwargs: torch.sin(1.3 * torch.arange(1, shape[-1] + 1, dtype=torch.float64))
cm = CMStochasticIterativeScheduler(clip_denoised=False)
cm.set_timesteps(2)
cm.sigmas = cm.sigmas.double()  # diffusers stores float32 sigmas; keep the step math in float64
cm_sample = X0 * 80.0
cm_output = 0.5 * cm_sample + C
c_skip, c_out = cm.get_scalings_for_boundary_condition(cm.sigmas[0].to(torch.float64))
cm_denoised = c_out * cm_output + c_skip * cm_sample
cm_next = cm.step(cm_output, cm.timesteps[0], cm_sample).prev_sample
print("CM sigma_next", f"{float(cm.sigmas[1]):.17g}", "denoised", fmt(cm_denoised), "renoised", fmt(cm_next))
