---
title: "LogSNRImportanceSampling<T>"
description: "Log-SNR importance sampling for efficient timestep selection during diffusion training."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Schedulers.NoiseSchedules`

Log-SNR importance sampling for efficient timestep selection during diffusion training.

## For Beginners

During training, some timesteps are more important than others
for the model to learn from. This utility picks timesteps that give the model the
most useful learning signal, making training more efficient.

## How It Works

Instead of uniformly sampling timesteps during training, samples proportionally to the
gradient magnitude at each timestep. Approximates this by sampling from a distribution
that is uniform in log-SNR space, focusing training on the most informative timesteps.

Reference: Hang et al., "Efficient Diffusion Training via Min-SNR Weighting Strategy", ICCV 2023

## Methods

| Method | Summary |
|:-----|:--------|
| `SampleTimesteps(Int32,Vector<>,Random)` | Samples timesteps with importance weighting based on log-SNR distribution. |

