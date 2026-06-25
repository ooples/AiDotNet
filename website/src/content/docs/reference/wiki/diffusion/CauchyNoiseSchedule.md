---
title: "CauchyNoiseSchedule<T>"
description: "Cauchy noise schedule using extremely heavy-tailed Cauchy distribution for noise sampling."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Schedulers.NoiseSchedules`

Cauchy noise schedule using extremely heavy-tailed Cauchy distribution for noise sampling.

## For Beginners

Cauchy noise has the heaviest tails — it produces extreme values
more often than Gaussian or Laplace noise. This can help the model generate very
sharp, high-contrast images but needs careful tuning.

## How It Works

Uses Cauchy-distributed noise which has even heavier tails than Laplace. This can
improve model robustness to outliers and enhance generation of extreme contrast regions.
Should be used carefully as Cauchy distribution has undefined mean and variance.

## Methods

| Method | Summary |
|:-----|:--------|
| `SampleNoise(Int32,Random)` | Generates Cauchy-distributed noise values. |

