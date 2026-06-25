---
title: "UniPCScheduler<T>"
description: "UniPC (Unified Predictor-Corrector) scheduler for fast, high-quality diffusion sampling."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Schedulers`

UniPC (Unified Predictor-Corrector) scheduler for fast, high-quality diffusion sampling.

## For Beginners

UniPC is like a "guess and check" approach to denoising.

Most schedulers just predict the next step (predict only):

- Step 1: Predict the cleaner image → use as-is

UniPC adds a correction step for better accuracy:

- Step 1: Predict the cleaner image (predictor step)
- Step 2: Check how good the prediction was and improve it (corrector step)

This two-phase approach means:

- Better quality at the same number of steps (e.g., 10-step UniPC ≈ 15-step DDIM)
- Or same quality with fewer steps (faster generation)

Key characteristics:

- Combines predictor-corrector methodology with multi-step methods
- Supports orders 1-3 (higher = more accurate per step)
- Deterministic by default
- Particularly effective at very low step counts (5-15 steps)

Used by:

- ComfyUI, A1111, and many Stable Diffusion UIs as an alternative scheduler
- Effective with SD 1.5, SDXL, and other diffusion models

## How It Works

UniPC combines predictor and corrector steps into a unified framework, achieving
superior sampling quality with fewer function evaluations compared to pure predictor
methods like DDIM or DPM-Solver++.

**Reference:** Zhao et al., "UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models", NeurIPS 2023

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `UniPCScheduler(SchedulerConfig<>,Int32,Boolean)` | Initializes a new instance of the UniPC scheduler. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeScheduleValues` | Computes lambda (log-SNR), alpha_t, and sigma_t values for the inference schedule. |
| `ConvertModelOutputToDataPrediction(Vector<>,Int32,Vector<>)` | Converts model output to data prediction (x_0 form) based on prediction type. |
| `CopyVector(Vector<>)` | Creates a copy of a vector. |
| `CorrectorStep(Vector<>,Vector<>,Int32,Int32,Int32)` | Corrector step: refines the predictor output using interpolation error estimation. |
| `FindTimestepIndex(Int32)` | Finds the index of a timestep in the current schedule. |
| `GetState` |  |
| `LoadState(Dictionary<String,Object>)` |  |
| `PredictorStep(Vector<>,Int32,Int32,Int32)` | Predictor step: multi-step extrapolation to estimate the next sample. |
| `ResetState` | Resets the scheduler state for a new generation run. |
| `SetTimesteps(Int32)` | Sets up the inference timesteps and computes lambda/sigma/alpha schedules. |
| `Step(Vector<>,Int32,Vector<>,,Vector<>)` | Performs one UniPC predictor-corrector denoising step. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_alphaTs` | Alpha_t values for inference timesteps. |
| `_dataPredictionHistory` | History of data predictions (x_0 form) for multi-step methods. |
| `_lambdas` | Lambda values (log-SNR) for each inference timestep. |
| `_sigmaTs` | Sigma_t values for inference timesteps. |
| `_solverOrder` | Maximum order of the predictor-corrector solver (1-3). |
| `_stepCounter` | Current step counter for tracking multi-step order. |
| `_useCorrectorStep` | Whether to apply the corrector step after the predictor. |

