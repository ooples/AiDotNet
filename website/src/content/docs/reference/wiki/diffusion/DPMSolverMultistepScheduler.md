---
title: "DPMSolverMultistepScheduler<T>"
description: "DPM-Solver++ multistep scheduler for fast diffusion model sampling."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Schedulers`

DPM-Solver++ multistep scheduler for fast diffusion model sampling.

## For Beginners

DPM-Solver++ is one of the fastest high-quality samplers.

Think of it like navigating with a GPS that remembers your path:

- Euler: Looks at current position only, takes simple steps
- DPM-Solver++: Remembers previous positions, predicts better next steps

Key characteristics:

- Very fast: Good quality in just 15-25 steps
- Multi-order: Uses 1st, 2nd, or 3rd order methods adaptively
- Deterministic: Same seed always gives same result
- Widely adopted: Used as default in many Stable Diffusion implementations

The "multistep" means it stores previous model outputs and uses them
to make more accurate predictions, similar to Adams-Bashforth methods
in numerical analysis.

## How It Works

DPM-Solver++ is a high-order ODE solver specifically designed for diffusion models.
It achieves state-of-the-art sampling quality with very few steps (10-25) by using
multi-step methods that leverage history of previous model evaluations.

**Reference:** Lu et al., "DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models", 2022

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DPMSolverMultistepScheduler(SchedulerConfig<>,Int32)` | Initializes a new instance of the DPM-Solver++ multistep scheduler. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeScheduleValues` | Computes lambda (log-SNR), alpha_t, and sigma_t values for the inference schedule. |
| `ConvertModelOutputToDataPrediction(Vector<>,Int32,Vector<>)` | Converts model output to data prediction (x_0 form) based on prediction type. |
| `CopyVector(Vector<>)` | Creates a copy of a vector. |
| `FindTimestepIndex(Int32)` | Finds the index of a timestep in the current schedule. |
| `FirstOrderUpdate(Vector<>,Int32,Int32)` | First-order DPM-Solver++ update (equivalent to DDIM). |
| `GetState` |  |
| `LoadState(Dictionary<String,Object>)` |  |
| `ResetState` | Resets the scheduler state for a new generation run. |
| `SecondOrderUpdate(Vector<>,Int32,Int32)` | Second-order DPM-Solver++ update using one previous model output. |
| `SetTimesteps(Int32)` | Sets up the inference timesteps and computes lambda/sigma/alpha schedules. |
| `Step(Vector<>,Int32,Vector<>,,Vector<>)` | Performs one DPM-Solver++ multistep denoising step. |
| `ThirdOrderUpdate(Vector<>,Int32,Int32)` | Third-order DPM-Solver++ update using two previous model outputs. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_alphaTs` | Alpha_t values for inference timesteps. |
| `_lambdas` | Lambda values (log-SNR) for each inference timestep. |
| `_modelOutputHistory` | History of model outputs for multi-step methods. |
| `_sigmaTs` | Sigma_t values for inference timesteps. |
| `_solverOrder` | Maximum order of the solver (1, 2, or 3). |
| `_stepCounter` | Current step counter for tracking multi-step order. |

