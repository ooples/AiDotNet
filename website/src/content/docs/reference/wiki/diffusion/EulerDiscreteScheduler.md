---
title: "EulerDiscreteScheduler<T>"
description: "Euler discrete scheduler for diffusion model sampling."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Schedulers`

Euler discrete scheduler for diffusion model sampling.

## For Beginners

The Euler scheduler is one of the most popular sampling methods.

Imagine you're navigating from point A (pure noise) to point B (clean image):

- DDPM: Takes tiny, cautious steps with some randomness
- Euler: Uses calculus (ODE solving) to take direct, efficient steps

Key characteristics:

- Deterministic: Same seed always produces the same result
- Fast convergence: Good results in 20-50 steps
- Simple and efficient: Low computational overhead per step
- Widely used in Stable Diffusion UIs (often called "Euler" or "Euler a")

The Euler method converts sigma (noise level) at each step and predicts
the "derivative" of the denoising trajectory, then takes a step along it.

## How It Works

The Euler scheduler implements first-order ODE solving for diffusion sampling.
It uses Euler's method to solve the probability flow ODE, providing fast
deterministic sampling with good quality at moderate step counts (20-50 steps).

**Reference:** Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models", NeurIPS 2022

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EulerDiscreteScheduler(SchedulerConfig<>)` | Initializes a new instance of the Euler discrete scheduler. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeSigmas` | Computes sigma values from the alpha cumulative product schedule. |
| `FindTimestepIndex(Int32)` | Finds the index of a timestep in the current schedule. |
| `SetTimesteps(Int32)` | Sets up the inference timesteps and computes sigma schedule. |
| `Step(Vector<>,Int32,Vector<>,,Vector<>)` | Performs one Euler discrete denoising step. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_sigmas` | Sigma values (noise levels) for each inference timestep. |

