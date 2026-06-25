---
title: "HeunDiscreteScheduler<T>"
description: "Heun discrete scheduler for diffusion model sampling using second-order Heun's method."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Schedulers`

Heun discrete scheduler for diffusion model sampling using second-order Heun's method.

## For Beginners

Heun's method is like Euler but smarter:

1. Euler takes one step and hopes for the best
2. Heun takes a trial step, evaluates the derivative there too,

then averages both derivatives for a more accurate step

Key characteristics:

- Second-order accuracy (better than Euler per step)
- Two model evaluations per step (so 20 Heun steps = 40 model calls)
- Good quality with fewer steps than Euler
- Deterministic: same seed always produces the same result

When to use Heun:

- When you want higher quality per step than Euler
- When model evaluation cost is acceptable
- When you want smooth, accurate trajectories

## How It Works

The Heun scheduler implements second-order ODE solving for diffusion sampling.
It performs two model evaluations per step (predictor + corrector) to achieve
higher accuracy than first-order methods like Euler.

**Reference:** Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models", NeurIPS 2022

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HeunDiscreteScheduler(SchedulerConfig<>)` | Initializes a new instance of the Heun discrete scheduler. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeSigmas` | Computes sigma values from the alpha cumulative product schedule. |
| `ConvertToPredOriginal(Vector<>,Vector<>,,Int32)` | Converts model output to predicted original sample based on prediction type. |
| `CreateDefault` | Creates a Heun scheduler with default Stable Diffusion settings. |
| `FindTimestepIndex(Int32)` | Finds the index of a timestep in the current schedule. |
| `HeunCorrectorStep(Vector<>,Vector<>)` | Second pass: use the second model evaluation at the intermediate point to compute the Heun corrector step. |
| `HeunPredictorStep(Vector<>,Int32,Vector<>)` | First pass: compute derivative d1, take Euler step, store state for corrector. |
| `SetTimesteps(Int32)` | Sets up the inference timesteps and computes sigma schedule. |
| `Step(Vector<>,Int32,Vector<>,,Vector<>)` | Performs one Heun denoising step (second-order method). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_sigmas` | Sigma values (noise levels) for each inference timestep. |

