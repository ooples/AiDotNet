---
title: "LMSDiscreteScheduler<T>"
description: "Linear Multi-Step (LMS) discrete scheduler for diffusion model sampling."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Schedulers`

Linear Multi-Step (LMS) discrete scheduler for diffusion model sampling.

## For Beginners

LMS uses memory of past steps to predict the future better:

Imagine navigating a winding road:

- Euler: Looks only at the current direction
- LMS: Remembers the last 4 turns and uses that pattern to predict the road ahead

Key characteristics:

- Multi-step method using derivative history (order 1-4)
- Better accuracy than single-step methods at the same cost
- One model evaluation per step
- Deterministic: same seed always produces the same result

The method computes Adams-Bashforth-style coefficients from the sigma schedule
and applies them to the stored derivative history.

## How It Works

The LMS scheduler uses a linear multi-step method to solve the diffusion ODE.
It maintains a history of previous derivatives and uses polynomial interpolation
to predict the next step more accurately.

**Reference:** Based on Adams-Bashforth multi-step ODE methods applied to diffusion.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LMSDiscreteScheduler(SchedulerConfig<>,Int32)` | Initializes a new instance of the LMS discrete scheduler. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeLMSCoefficients(Int32,Int32)` | Computes Adams-Bashforth-style coefficients for the LMS method. |
| `CreateDefault` | Creates an LMS scheduler with default Stable Diffusion settings. |
| `GetAdamsBashforthWeight(Int32,Int32)` | Gets the Adams-Bashforth coefficient for a given order and index. |
| `SetTimesteps(Int32)` |  |
| `Step(Vector<>,Int32,Vector<>,,Vector<>)` | Performs one LMS denoising step using stored derivative history. |

