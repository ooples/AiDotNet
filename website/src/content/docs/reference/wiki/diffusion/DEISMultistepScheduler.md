---
title: "DEISMultistepScheduler<T>"
description: "Diffusion Exponential Integrator Sampler (DEIS) for fast diffusion model sampling."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Schedulers`

Diffusion Exponential Integrator Sampler (DEIS) for fast diffusion model sampling.

## For Beginners

DEIS is a smart math-based sampler:

The key insight is that the diffusion ODE has an exponential structure.
By using exponential integrators (which are exact for exponential functions),
DEIS captures more of the solution's behavior in fewer steps.

Key characteristics:

- Multi-step method (orders 1-3)
- Uses exponential integrator formulas
- Excellent quality at low step counts (10-20 steps)
- Deterministic: same seed always produces the same result
- Stores derivative history like LMS, but uses exponential interpolation

Think of it like: instead of approximating a curve with straight lines (Euler)
or polynomials (LMS), DEIS uses exponentials which better match the
diffusion process's natural shape.

## How It Works

DEIS uses exponential integrators with polynomial extrapolation to solve
the diffusion ODE. It achieves high-quality samples with very few steps
by leveraging the exponential structure of the diffusion process.

**Reference:** Zhang and Chen, "Fast Sampling of Diffusion Models with Exponential Integrator", ICLR 2023

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DEISMultistepScheduler(SchedulerConfig<>,Int32)` | Initializes a new instance of the DEIS multistep scheduler. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeLambda()` | Computes lambda = log(alpha/sigma) = -log(sigma) for the sigma parameterization. |
| `CreateDefault` | Creates a DEIS scheduler with default Stable Diffusion settings. |
| `SetTimesteps(Int32)` |  |
| `Step(Vector<>,Int32,Vector<>,,Vector<>)` | Performs one DEIS denoising step using exponential integrator formulas. |

