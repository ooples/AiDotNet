---
title: "DiffusionNoiseHelper<T>"
description: "Helper class for noise sampling operations in diffusion models."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Helpers`

Helper class for noise sampling operations in diffusion models.

## For Beginners

Diffusion models work by adding and removing noise from data.
This helper provides the mathematical operations needed for that process:

- Sampling Gaussian (bell-curve) noise
- Computing noise schedules
- Scaling noise for different timesteps

## How It Works

This static helper provides common noise sampling operations used throughout diffusion models,
ensuring consistent implementations and avoiding code duplication.

## Methods

| Method | Summary |
|:-----|:--------|
| `AddNoise(Tensor<>,Tensor<>,,)` | Adds noise to a signal at a specified timestep using the scheduler's noise schedule. |
| `BoxMullerTransform(Random)` | Box-Muller transform to convert uniform random numbers to Gaussian. |
| `ComputeSNR()` | Computes the signal-to-noise ratio (SNR) for a given timestep. |
| `ComputeTimestepEmbedding(Int32,Int32)` | Computes sinusoidal embedding for a single timestep. |
| `ComputeTimestepEmbeddings(Int32[],Int32)` | Computes sinusoidal timestep embeddings (like in Transformers). |
| `LerpNoise(Tensor<>,Tensor<>,Double)` | Linearly interpolates between two noise tensors. |
| `SampleGaussian(Int32[],Nullable<Int32>)` | Samples Gaussian noise from a standard normal distribution N(0, 1). |
| `SampleGaussian(Int32[],Random)` | Samples Gaussian noise using a provided random number generator. |
| `SampleGaussianVector(Int32,Nullable<Int32>)` | Samples Gaussian noise as a Vector. |
| `SampleGaussianVector(Int32,Random)` | Samples Gaussian noise as a Vector using a provided random number generator. |
| `ScaleNoise(Tensor<>,Double)` | Scales noise by a given factor. |
| `SlerpNoise(Tensor<>,Tensor<>,Double)` | Spherical linear interpolation between two noise tensors. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Provides numeric operations for the specific type T. |

