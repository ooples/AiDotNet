---
title: "MisGANOptions<T>"
description: "Configuration options for MisGAN, a GAN for learning from incomplete data with dual generator/discriminator pairs for data and missingness patterns."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for MisGAN, a GAN for learning from incomplete data with
dual generator/discriminator pairs for data and missingness patterns.

## For Beginners

MisGAN handles datasets with missing values:

Real datasets often have missing values (e.g., patients who skip tests).
MisGAN learns both:

1. What the data looks like when complete
2. Which values tend to be missing (and why)

This produces synthetic data with realistic patterns of completeness.

Example:

## How It Works

MisGAN uses four networks:

- **Data generator**: Generates complete data samples
- **Mask generator**: Generates realistic missingness patterns
- **Data discriminator**: Judges realism of data values
- **Mask discriminator**: Judges realism of missingness patterns

Reference: "MisGAN: Learning from Incomplete Data with GANs" (Li et al., ICLR 2019)

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the training batch size. |
| `DiscriminatorDropout` | Gets or sets the dropout rate for discriminator hidden layers. |
| `DiscriminatorSteps` | Gets or sets the number of discriminator training steps per generator step. |
| `EmbeddingDimension` | Gets or sets the noise dimension. |
| `Epochs` | Gets or sets the number of training epochs. |
| `GradientPenaltyWeight` | Gets or sets the weight for the WGAN-GP gradient penalty term. |
| `HiddenDimensions` | Gets or sets the hidden layer sizes. |
| `LearningRate` | Gets or sets the learning rate. |
| `MissingRate` | Gets or sets the expected rate of missing values in the data. |
| `VGMModes` | Gets or sets the number of VGM modes for continuous column encoding. |

