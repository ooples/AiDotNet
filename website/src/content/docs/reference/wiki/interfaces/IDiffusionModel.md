---
title: "IDiffusionModel<T>"
description: "Interface for diffusion-based generative models."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for diffusion-based generative models.

## For Beginners

Diffusion models are like learning to reverse a process of adding static to a TV signal.

How diffusion works:

1. Forward process (training): Start with real data, gradually add noise until it's pure static
2. Reverse process (generation): Start with pure static, gradually remove noise to create new data

The model learns: "Given this noisy version, what did the original look like?"

This is different from other generative models:

- GANs: Two networks competing (generator vs discriminator)
- VAEs: Compress and decompress through a bottleneck
- Diffusion: Iteratively denoise from random noise

Diffusion models are known for:

- High quality outputs (often better than GANs)
- Stable training (no mode collapse)
- Good diversity (produces varied outputs)
- Slower generation (many denoising steps needed)

## How It Works

Diffusion models are a class of generative models that learn to create data by reversing
a gradual noising process. They have achieved state-of-the-art results in image generation,
audio synthesis, and other generative tasks.

**Key components:**

- Noise prediction model: A neural network that predicts noise in images
- Noise scheduler: Controls the noise schedule (see `INoiseScheduler`)
- Loss function: Measures how well the model predicts noise (usually MSE)

This interface extends `IFullModel` to provide a consistent API
for diffusion models while inheriting all the standard model capabilities (training, saving,
loading, gradients, checkpointing, etc.).

## Properties

| Property | Summary |
|:-----|:--------|
| `Scheduler` | Gets the step scheduler used for the diffusion process. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeLoss(Tensor<>,Tensor<>,Int32[])` | Computes the training loss for a batch of samples. |
| `Generate(Int32[],Int32,Nullable<Int32>)` | Generates samples by iteratively denoising from random noise. |
| `PredictNoise(Tensor<>,Int32)` | Predicts the noise in a noisy sample at a given timestep. |

