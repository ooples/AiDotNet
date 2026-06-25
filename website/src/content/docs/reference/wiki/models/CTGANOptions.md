---
title: "CTGANOptions<T>"
description: "Configuration options for CTGAN (Conditional Tabular GAN), a generative adversarial network specifically designed for generating realistic synthetic tabular data."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for CTGAN (Conditional Tabular GAN), a generative adversarial network
specifically designed for generating realistic synthetic tabular data.

## For Beginners

CTGAN is like having two neural networks compete to create fake data:

- The **Generator** creates fake rows of data from random noise
- The **Discriminator** tries to distinguish real rows from fake ones
- As they compete, the generator learns to produce increasingly realistic data

Special features for tabular data:

- Handles mixed data types (numbers and categories) in the same table
- Uses conditional generation to ensure rare categories are well-represented
- Uses Wasserstein loss with gradient penalty for stable training

Example:

## How It Works

CTGAN uses a conditional GAN architecture with several innovations for tabular data:

- Variational Gaussian Mixture (VGM) mode-specific normalization for continuous columns
- Training-by-sampling with conditional vectors for handling imbalanced categories
- WGAN-GP (Wasserstein GAN with Gradient Penalty) for stable training
- PacGAN packing to prevent mode collapse

Reference: "Modeling Tabular Data using Conditional GAN" (Xu et al., NeurIPS 2019)

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

