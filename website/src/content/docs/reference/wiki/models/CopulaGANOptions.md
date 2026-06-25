---
title: "CopulaGANOptions<T>"
description: "Configuration options for CopulaGAN, a synthetic tabular data generator that combines Gaussian copula transformations with the CTGAN training pipeline."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for CopulaGAN, a synthetic tabular data generator that combines
Gaussian copula transformations with the CTGAN training pipeline.

## For Beginners

CopulaGAN improves on CTGAN by "normalizing" each numerical column
before training. Think of it like this:

1. **Copula Transform**: Each number column is reshaped to look like a bell curve

(Gaussian distribution) using a mathematical trick called a "copula."

2. **CTGAN Training**: The standard CTGAN generator/discriminator pipeline trains

on this nicely-shaped data, which is easier to learn.

3. **Inverse Transform**: Generated data is converted back to the original

distribution shapes.

This often produces better results than plain CTGAN for columns with unusual distributions
(e.g., heavily skewed income data, bimodal distributions).

Example:

## How It Works

CopulaGAN extends CTGAN by first transforming continuous columns using Gaussian copulas.
This maps each continuous column's distribution to a standard normal via CDF-then-quantile,
making the CTGAN generator's job easier and often improving generation quality for
columns with complex or skewed distributions.

Reference: "Synthesizing Tabular Data using Copulas" (2020)

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the training batch size. |
| `DiscriminatorDimensions` | Gets or sets the hidden layer sizes for the discriminator network. |
| `DiscriminatorDropout` | Gets or sets the dropout rate for the discriminator. |
| `DiscriminatorSteps` | Gets or sets the number of discriminator training steps per generator step. |
| `EmbeddingDimension` | Gets or sets the dimension of the random noise vector fed to the generator. |
| `Epochs` | Gets or sets the number of training epochs. |
| `GeneratorDimensions` | Gets or sets the hidden layer sizes for the generator network. |
| `GradientPenaltyWeight` | Gets or sets the gradient penalty weight (lambda) for WGAN-GP. |
| `LearningRate` | Gets or sets the learning rate for both generator and discriminator optimizers. |
| `PacSize` | Gets or sets the PacGAN packing size. |
| `VGMModes` | Gets or sets the number of Gaussian mixture components for VGM normalization. |

