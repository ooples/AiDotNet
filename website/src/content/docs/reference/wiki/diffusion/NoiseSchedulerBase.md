---
title: "NoiseSchedulerBase<T>"
description: "Base class for diffusion model noise schedulers providing common functionality."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Diffusion.Schedulers`

Base class for diffusion model noise schedulers providing common functionality.

## For Beginners

This is the foundation that all noise schedulers build upon.
It handles the common math and state management that every scheduler needs:

- Computing the noise schedule (how much noise at each step)
- Tracking the current state for saving/loading
- Adding noise during training

Specific schedulers like DDIM, PNDM, and DPM-Solver extend this base to implement
their unique denoising strategies.

## How It Works

This abstract base class implements the common behavior for all noise schedulers,
including beta schedule computation, alpha cumulative product calculation, noise addition,
and state management for checkpointing.

**Note:** This class was renamed from StepSchedulerBase to NoiseSchedulerBase to avoid
confusion with learning rate schedulers. Noise schedulers are specific to diffusion models.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NoiseSchedulerBase(SchedulerConfig<>)` | Initializes a new instance of the NoiseSchedulerBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Config` | Gets the configuration options for the scheduler. |
| `Engine` | Gets the compute engine for GPU-accelerated vectorized operations. |
| `Timesteps` |  |
| `TrainTimesteps` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddNoise(Vector<>,Vector<>,Int32)` |  |
| `ClipSampleIfNeeded(Vector<>)` | Clips sample values to [-1, 1] if configured. |
| `ComputeAlphas` | Computes alpha values and their cumulative products from betas. |
| `GetAlphaCumulativeProduct(Int32)` |  |
| `GetState` |  |
| `InitializeBetaSchedule` | Initializes the beta schedule based on the configuration. |
| `InitializeLinearBetaSchedule(Int32)` | Initializes a linear beta schedule. |
| `InitializeScaledLinearBetaSchedule(Int32)` | Initializes a scaled linear beta schedule (used by Stable Diffusion). |
| `InitializeSquaredCosineBetaSchedule(Int32)` | Initializes a squared cosine beta schedule (improved schedule). |
| `LoadState(Dictionary<String,Object>)` |  |
| `SetTimestepArray(Int32[])` | Sets the timestep array directly. |
| `SetTimesteps(Int32)` |  |
| `Step(Vector<>,Int32,Vector<>,,Vector<>)` |  |
| `ValidateStepParameters(Vector<>,Vector<>,Int32)` | Validates common step parameters. |

## Fields

| Field | Summary |
|:-----|:--------|
| `Alphas` | Alpha values (1 - beta) representing signal retention at each timestep. |
| `AlphasCumulativeProduct` | Cumulative product of alphas representing total signal retention at each timestep. |
| `Betas` | Beta values (noise variance) at each training timestep. |
| `NumOps` | Provides numeric operations for the specific type T. |
| `_timesteps` | The timesteps for the current inference schedule. |

