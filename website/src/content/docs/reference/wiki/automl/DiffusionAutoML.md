---
title: "DiffusionAutoML<T>"
description: "AutoML for diffusion models with automatic hyperparameter optimization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AutoML`

AutoML for diffusion models with automatic hyperparameter optimization.

## For Beginners

This class automatically finds the best settings for your diffusion model.

When using diffusion models, there are many choices to make:

- What type of neural network architecture (U-Net, DiT, etc.)
- What sampling scheduler (DDIM, Euler, DPM-Solver, etc.)
- How many inference steps to use
- What guidance scale for conditional generation
- Training hyperparameters like learning rate

DiffusionAutoML tries different combinations automatically and finds
what works best for your specific data and use case.

## How It Works

DiffusionAutoML automatically searches for optimal diffusion model configurations,
including noise predictor architecture, scheduler type, and training hyperparameters.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DiffusionAutoML(Nullable<Int32>)` | Initializes a new instance of the DiffusionAutoML class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BestConfig` | Gets the best diffusion configuration found during search. |
| `NoisePredictorTypesToTry` | Gets the list of noise predictor types to try during search. |
| `SchedulerTypesToTry` | Gets the list of scheduler types to try during search. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateInstanceForCopy` | Creates an instance for deep copy. |
| `CreateModelAsync(Type,Dictionary<String,Object>)` | Creates a diffusion model based on the specified parameters. |
| `GetDefaultSearchSpace(Type)` | Gets the default search space for diffusion models. |
| `SearchAsync(Tensor<>,Tensor<>,Tensor<>,Tensor<>,TimeSpan,CancellationToken)` | Searches for the best diffusion model configuration. |
| `SuggestNextTrialAsync` | Suggests the next trial parameters based on search history. |

