---
title: "LaplaceNoiseSchedule<T>"
description: "Laplace noise schedule using heavy-tailed Laplace distribution for noise sampling."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Schedulers.NoiseSchedules`

Laplace noise schedule using heavy-tailed Laplace distribution for noise sampling.

## For Beginners

Laplace noise has more "extreme" values than standard Gaussian
noise. Using it during training helps the model handle sharp edges and high-contrast
areas better, leading to crisper generated images.

## How It Works

Replaces Gaussian noise with Laplace-distributed noise during the forward process.
The heavier tails of the Laplace distribution help the model handle extreme values
better, improving generation of high-contrast and high-frequency details.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LaplaceNoiseSchedule(Double)` | Initializes a new instance with the specified Laplace scale parameter. |

## Methods

| Method | Summary |
|:-----|:--------|
| `SampleNoise(Int32,Random)` | Generates Laplace-distributed noise values. |

