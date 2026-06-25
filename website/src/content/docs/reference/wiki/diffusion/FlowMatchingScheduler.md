---
title: "FlowMatchingScheduler<T>"
description: "Flow matching scheduler implementing rectified flow ODE sampling."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Schedulers`

Flow matching scheduler implementing rectified flow ODE sampling.

## For Beginners

Flow matching works like drawing a straight line from noise to image.

Traditional diffusion (DDPM/DDIM):

- Adds noise using: x_t = sqrt(alpha) * image + sqrt(1-alpha) * noise
- The path from noise to image is curved (follows a complex schedule)
- Needs many steps to follow the curved path accurately

Flow matching (rectified flow):

- Uses simple linear interpolation: x_t = (1-t) * image + t * noise
- The path from noise to image is a straight line
- Can traverse the straight path in fewer steps (often 20-50 vs 50-100+)
- The model predicts velocity v = noise - image (direction to move)

ODE sampling step:

- Start at x_1 (pure noise)
- At each step: x_{t-dt} = x_t - dt * v(x_t, t)
- After all steps: arrive at x_0 (clean image)

Used by:

- Stable Diffusion 3 (SD3)
- FLUX.1
- Stable Diffusion 3.5

## How It Works

Flow matching is a fundamentally different approach from DDPM-style diffusion. Instead of
learning to predict noise (epsilon) at each step, the model learns a velocity field v(x_t, t)
that defines an ordinary differential equation (ODE) transporting data between noise and signal.

**Reference:** Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023;
Liu et al., "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow", ICLR 2023

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FlowMatchingScheduler(SchedulerConfig<>)` | Initializes a new flow matching scheduler with rectified flow defaults. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddNoise(Vector<>,Vector<>,Int32)` | Adds noise using flow matching linear interpolation. |
| `ComputeSigmas(Int32)` | Computes sigma values for optional stochastic sampling. |
| `ComputeTimeValues(Int32)` | Pre-computes continuous time values for all training timesteps. |
| `CreateDefault` | Creates a flow matching scheduler with default SD3/FLUX configuration. |
| `GetContinuousTime(Int32)` | Converts an integer timestep to continuous time t in [0, 1]. |
| `GetPreviousContinuousTime(Int32)` | Gets the continuous time for the previous timestep in the inference schedule. |
| `GetState` |  |
| `SetTimesteps(Int32)` | Sets up linearly-spaced inference timesteps for flow matching. |
| `Step(Vector<>,Int32,Vector<>,,Vector<>)` | Performs one flow matching ODE step (Euler method). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_sigmas` | Sigma values for each timestep (optional noise scaling for stochastic sampling). |
| `_timeValues` | The continuous time values corresponding to the discrete timesteps. |

