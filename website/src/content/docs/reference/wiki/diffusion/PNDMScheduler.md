---
title: "PNDMScheduler<T>"
description: "PNDM (Pseudo Numerical Methods for Diffusion Models) scheduler implementation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Schedulers`

PNDM (Pseudo Numerical Methods for Diffusion Models) scheduler implementation.

## For Beginners

PNDM is an advanced method for fast image generation.

Think of diffusion like walking down a mountain (from noise to clean image):

- DDPM: Take 1000 tiny careful steps
- DDIM: Take 50 medium steps
- PNDM: Take 20 smart steps using "momentum" from previous steps

The key insight is that PNDM remembers its previous predictions and uses them
to make better guesses about where to step next. It's like a ball rolling down
a hill - it uses its momentum to move faster.

Advantages:

- Very fast generation (often 20-25 steps for good quality)
- Uses multi-step methods from numerical analysis
- Good balance of speed and quality

The scheduler operates in two phases:

1. Prk (Pseudo Runge-Kutta) for initial steps
2. Plms (Pseudo Linear Multi-Step) for remaining steps

## How It Works

PNDM uses pseudo numerical methods to accelerate diffusion sampling. It can achieve
high-quality results with even fewer steps than DDIM by using a combination of
linear multi-step methods and improved transfer techniques.

**Reference:** "Pseudo Numerical Methods for Diffusion Models on Manifolds" by Liu et al., 2022

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PNDMScheduler(SchedulerConfig<>)` | Initializes a new instance of the PNDM scheduler. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeLinearMultiStep` | Computes the linear multi-step combination of model outputs (VECTORIZED). |
| `ComputePrevSample(Vector<>,Vector<>,)` | Computes the previous sample from predicted original and model output (VECTORIZED). |
| `CopyVector(Vector<>)` | Creates a copy of a vector. |
| `GetState` |  |
| `LoadState(Dictionary<String,Object>)` |  |
| `PredictOriginalSample(Vector<>,Int32,Vector<>)` | Predicts the original sample from noise prediction (VECTORIZED). |
| `ResetState` | Resets the scheduler state for a new generation run. |
| `SetTimesteps(Int32)` | Sets up the inference timesteps and resets the scheduler state. |
| `Step(Vector<>,Int32,Vector<>,,Vector<>)` | Performs one PNDM denoising step. |
| `StepPlms(Vector<>,Int32,Vector<>)` | Performs a pseudo linear multi-step (plms) step. |
| `StepPrk(Vector<>,Int32,Vector<>)` | Performs a pseudo Runge-Kutta step (warmup phase). |

## Fields

| Field | Summary |
|:-----|:--------|
| `PrkModeSteps` | Number of warmup steps before switching to linear multi-step. |
| `_counter` | Current step counter within a single inference run. |
| `_currentSample` | Current sample being processed (for Runge-Kutta steps). |
| `_ets` | History of model outputs for multi-step methods. |

