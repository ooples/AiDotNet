---
title: "DDPMModel<T>"
description: "DDPM (Denoising Diffusion Probabilistic Models) implementation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion`

DDPM (Denoising Diffusion Probabilistic Models) implementation.

## For Beginners

DDPM is like learning to restore a damaged photograph.

Training process:

1. Take a clear photo
2. Add a specific amount of noise (determined by timestep)
3. Train a neural network to predict what noise was added
4. Repeat with different photos and noise levels

Generation process:

1. Start with pure random noise
2. Ask the trained model "what noise is in this?"
3. Remove the predicted noise to get a slightly clearer image
4. Repeat 1000 times (or use DDIM for faster generation)
5. End up with a new, never-before-seen image

This implementation provides a minimal but functional DDPM that serves as:

- A reference implementation for understanding diffusion
- A base for more sophisticated diffusion models
- A demonstration of scheduler integration

## How It Works

DDPM is the foundational diffusion model architecture that introduced the modern
approach to diffusion-based generation. It learns to reverse a gradual noising
process to generate data from pure noise.

**Reference:** "Denoising Diffusion Probabilistic Models" by Ho et al., 2020

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DDPMModel(INoiseScheduler<>,Func<Tensor<>,Int32,Tensor<>>)` | Initializes a new instance of the DDPM model with a scheduler only. |
| `DDPMModel(Int32)` | Initializes a new instance of the DDPM model with a seed for reproducibility. |
| `DDPMModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,Func<Tensor<>,Int32,Tensor<>>,Int32,Int32,Nullable<Int32>)` | Initializes a new instance of the DDPM model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `Create(SchedulerConfig<>,Func<Tensor<>,Int32,Tensor<>>)` | Creates a DDPM model with a custom noise predictor function (for testing). |
| `Create(SchedulerConfig<>,UNetNoisePredictor<>)` | Creates a DDPM model with a custom scheduler configuration. |
| `DeepCopy` |  |
| `GetParameters` |  |
| `PredictNoise(Tensor<>,Int32)` |  |
| `SetParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_customPredictor` | Optional custom noise prediction function for testing or custom architectures. |
| `_unet` | The UNet noise predictor per Ho et al. |

