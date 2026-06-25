---
title: "EulerAncestralDiscreteScheduler<T>"
description: "Euler Ancestral discrete scheduler for diffusion model sampling."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Schedulers`

Euler Ancestral discrete scheduler for diffusion model sampling.

## For Beginners

This is the "Euler a" sampler commonly seen in Stable Diffusion UIs.

The difference from regular Euler:

- Euler: Deterministic - same seed always gives identical results
- Euler Ancestral: Stochastic - adds controlled randomness at each step

This stochasticity means:

- More creative/diverse outputs for the same prompt
- Slightly less reproducible (even with same seed, small changes cascade)
- Often produces more detailed, painterly results
- Popular choice for artistic/creative generation

The "ancestral" part means it samples from the reverse diffusion posterior,
similar to how DDPM adds noise at each step, but using Euler integration
for the deterministic part.

## How It Works

The Euler Ancestral scheduler combines Euler's method with ancestral sampling,
adding stochastic noise at each step. This creates more diverse outputs compared
to the deterministic Euler scheduler, at the cost of slightly less consistency.

**Reference:** Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models", NeurIPS 2022

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EulerAncestralDiscreteScheduler(SchedulerConfig<>)` | Initializes a new instance of the Euler Ancestral discrete scheduler. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeSigmas` | Computes sigma values from the alpha cumulative product schedule. |
| `FindTimestepIndex(Int32)` | Finds the index of a timestep in the current schedule. |
| `SetTimesteps(Int32)` | Sets up the inference timesteps and computes sigma schedule. |
| `Step(Vector<>,Int32,Vector<>,,Vector<>)` | Performs one Euler Ancestral denoising step. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_sigmas` | Sigma values (noise levels) for each inference timestep. |

