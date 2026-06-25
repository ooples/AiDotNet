---
title: "DPCTGANOptions<T>"
description: "Configuration options for DP-CTGAN, a differentially private version of CTGAN that provides formal privacy guarantees while generating synthetic tabular data."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for DP-CTGAN, a differentially private version of CTGAN that provides
formal privacy guarantees while generating synthetic tabular data.

## For Beginners

DP-CTGAN generates fake data while mathematically guaranteeing
that the synthetic data doesn't reveal too much about any individual in the real data.

Think of it like CTGAN with a "privacy filter":

1. During training, gradients are clipped (bounded) so no single person's data

has too much influence on the model

2. Random noise is added to further obscure individual contributions
3. A "privacy budget" (epsilon) tracks how much privacy has been spent

Lower epsilon = more privacy but lower data quality.
Typical values: epsilon 1-10 for reasonable utility.

Example:

## How It Works

DP-CTGAN adds differential privacy to CTGAN through:

- **Per-sample gradient clipping**: Bounds the sensitivity of each training sample
- **Gaussian noise injection**: Adds calibrated noise to clipped gradients
- **Privacy accountant**: Tracks cumulative privacy loss (epsilon)

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the training batch size. |
| `ClipNorm` | Gets or sets the L2 norm clipping bound for per-sample gradients. |
| `Delta` | Gets or sets the delta parameter for (epsilon, delta)-differential privacy. |
| `DiscriminatorDimensions` | Gets or sets the hidden layer sizes for the discriminator. |
| `DiscriminatorDropout` | Gets or sets the discriminator dropout rate. |
| `DiscriminatorSteps` | Gets or sets the number of discriminator steps per generator step. |
| `EmbeddingDimension` | Gets or sets the dimension of the random noise vector. |
| `Epochs` | Gets or sets the number of training epochs. |
| `Epsilon` | Gets or sets the total privacy budget (epsilon) for the training process. |
| `GeneratorDimensions` | Gets or sets the hidden layer sizes for the generator. |
| `GradientPenaltyWeight` | Gets or sets the gradient penalty weight. |
| `LearningRate` | Gets or sets the learning rate. |
| `NoiseMultiplier` | Gets or sets the noise multiplier for Gaussian mechanism. |
| `PacSize` | Gets or sets the PacGAN packing size. |
| `VGMModes` | Gets or sets the number of VGM modes. |

