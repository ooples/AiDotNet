---
title: "DPMSolverSDEScheduler<T>"
description: "DPM++ 2M SDE scheduler — stochastic variant of DPM-Solver++ multistep."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Schedulers`

DPM++ 2M SDE scheduler — stochastic variant of DPM-Solver++ multistep.

## For Beginners

This is the "SDE Karras" sampler popular in Stable Diffusion UIs:

- DPM++ 2M: Fast, deterministic, same seed = same image
- DPM++ 2M SDE: Adds slight randomness for more creative/diverse results

Key characteristics:

- Stochastic: Adds controlled noise at each step
- Two-step method with previous derivative memory
- Popular in community UIs (often labeled "DPM++ 2M SDE Karras")
- Good diversity-quality tradeoff
- Works well with 20-30 steps

The SDE noise strength is controlled by the eta parameter.
Higher eta = more stochastic = more diverse but potentially lower quality.

## How It Works

DPM++ 2M SDE adds stochastic noise injection to the DPM-Solver++ 2M method.
This creates more diverse outputs while maintaining the efficiency of the
deterministic variant. The SDE formulation adds controlled randomness at each step.

**Reference:** Lu et al., "DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models", NeurIPS 2022

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DPMSolverSDEScheduler(SchedulerConfig<>,Nullable<Int32>)` | Initializes a new instance of the DPM++ 2M SDE scheduler. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateDefault(Nullable<Int32>)` | Creates a DPM++ 2M SDE scheduler with default Stable Diffusion settings. |
| `GenerateNoise(Int32)` | Generates Gaussian noise with the scheduler's random state. |
| `SetTimesteps(Int32)` |  |
| `Step(Vector<>,Int32,Vector<>,,Vector<>)` | Performs one DPM++ 2M SDE denoising step with stochastic noise injection. |

