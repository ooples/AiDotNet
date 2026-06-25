---
title: "CTABGANPlusOptions<T>"
description: "Configuration options for CTAB-GAN+, an enhanced conditional tabular GAN with auxiliary classifier discriminator and mixed-type encoder for high-quality synthetic data generation."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for CTAB-GAN+, an enhanced conditional tabular GAN with auxiliary
classifier discriminator and mixed-type encoder for high-quality synthetic data generation.

## For Beginners

CTAB-GAN+ is an improved version of CTGAN that produces
higher-quality synthetic data by using smarter training signals:

1. **Better discriminator**: Not only decides real/fake, but also classifies data categories
2. **Better encoding**: Handles rare categories better with log-frequency encoding
3. **Better evaluation**: Checks if generated data is useful for downstream ML tasks

Example:

## How It Works

CTAB-GAN+ extends CTGAN with several architectural improvements:

- **Auxiliary Classifier GAN (ACGAN)**: Discriminator also predicts class labels
- **Mixed-type encoder**: Log-frequency encoding for long-tail categoricals
- **Downstream losses**: Additional classification/regression losses for utility
- **Conditional vector**: Same training-by-sampling as CTGAN

Reference: "CTAB-GAN: Effective Table Data Synthesizing" (Zhao et al., 2021)

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the training batch size. |
| `ClassifierWeight` | Gets or sets the weight for the auxiliary classifier loss in the discriminator. |
| `DiscriminatorDimensions` | Gets or sets the hidden layer sizes for the discriminator network. |
| `DiscriminatorDropout` | Gets or sets the dropout rate for the discriminator. |
| `DiscriminatorSteps` | Gets or sets the number of discriminator training steps per generator step. |
| `EmbeddingDimension` | Gets or sets the dimension of the random noise vector fed to the generator. |
| `Epochs` | Gets or sets the number of training epochs. |
| `GeneratorDimensions` | Gets or sets the hidden layer sizes for the generator network. |
| `GradientPenaltyWeight` | Gets or sets the gradient penalty weight for WGAN-GP. |
| `InformationWeight` | Gets or sets the weight for the information loss (statistical similarity). |
| `LearningRate` | Gets or sets the learning rate for both generator and discriminator. |
| `PacSize` | Gets or sets the PacGAN packing size. |
| `TargetColumnIndex` | Gets or sets the index of the target/label column for the auxiliary classifier. |
| `VGMModes` | Gets or sets the number of VGM modes for continuous column normalization. |

