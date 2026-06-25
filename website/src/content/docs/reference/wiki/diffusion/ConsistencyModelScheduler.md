---
title: "ConsistencyModelScheduler<T>"
description: "Consistency Model scheduler for single-step or few-step diffusion sampling."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Schedulers`

Consistency Model scheduler for single-step or few-step diffusion sampling.

## For Beginners

Consistency models are the fastest type of diffusion model:

Normal diffusion: Start with noise, take 20-50 small steps to get a clean image
Consistency model: Start with noise, jump directly to the clean image in 1 step!

Key characteristics:

- Single-step generation possible (fastest diffusion method)
- Multi-step mode (2-4 steps) improves quality
- Maps noisy samples directly to clean data predictions
- Works with both distilled and directly trained consistency models

How multi-step consistency works:

1. Start with pure noise at sigma_max
2. Apply consistency function → get approximate clean image
3. Add noise back at a lower sigma level
4. Apply consistency function again → better clean image
5. Repeat for desired number of steps

This "denoise-then-add-noise" cycle progressively refines the output.

## How It Works

Consistency Models map any point on the probability flow ODE trajectory
directly to the trajectory's origin (the clean data). This allows single-step
generation, with optional multi-step refinement for higher quality.

**Reference:** Song et al., "Consistency Models", ICML 2023

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ConsistencyModelScheduler(SchedulerConfig<>,Double,Double,Nullable<Int32>)` | Initializes a new instance of the Consistency Model scheduler. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeSigmas(Int32)` | Computes sigma schedule for consistency model multi-step sampling. |
| `CreateDefault(Nullable<Int32>)` | Creates a Consistency Model scheduler with default settings. |
| `SetTimesteps(Int32)` |  |
| `Step(Vector<>,Int32,Vector<>,,Vector<>)` | Performs one consistency model denoising step. |

