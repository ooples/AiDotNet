---
title: "DPMSolverSinglestepScheduler<T>"
description: "DPM++ 2S Ancestral scheduler — single-step DPM-Solver++ with ancestral sampling."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Schedulers`

DPM++ 2S Ancestral scheduler — single-step DPM-Solver++ with ancestral sampling.

## For Beginners

This sampler combines accuracy with diversity:

- "2S": Two sub-steps within each step for second-order accuracy
- "Ancestral": Adds random noise at each step for diversity (like Euler Ancestral)

Key characteristics:

- Second-order accuracy without needing history from previous steps
- Stochastic (ancestral): different results each run unless seeded
- Good for creative/artistic generation where diversity is valued
- Works well with 20-30 steps

The "ancestral" noise means each step adds a small amount of noise,
making the trajectory stochastic. This can produce more diverse results
but may be less consistent than deterministic samplers.

## How It Works

DPM++ 2S a (single-step, ancestral) performs a second-order DPM-Solver step
within each timestep and adds ancestral noise for diversity. Unlike DPM++ 2M which
uses multi-step history, this computes the second-order update in a single step
by performing two sub-steps internally.

**Reference:** Lu et al., "DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models", NeurIPS 2022

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DPMSolverSinglestepScheduler(SchedulerConfig<>,Nullable<Int32>)` | Initializes a new instance of the DPM++ 2S ancestral scheduler. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateDefault(Nullable<Int32>)` | Creates a DPM++ 2S scheduler with default Stable Diffusion settings. |
| `SetTimesteps(Int32)` |  |
| `Step(Vector<>,Int32,Vector<>,,Vector<>)` | Performs one DPM++ 2S ancestral denoising step. |

