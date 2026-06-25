---
title: "TableGANOptions<T>"
description: "Configuration options for TableGAN, a DCGAN-style generative adversarial network for synthesizing tabular data with classification and information loss regularization."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for TableGAN, a DCGAN-style generative adversarial network for
synthesizing tabular data with classification and information loss regularization.

## For Beginners

TableGAN generates fake tabular data using three quality checks:

1. "Does it look real?" — the discriminator judges overall realism
2. "Are the labels correct?" — a classifier checks that labels match the data
3. "Are the statistics right?" — mean/variance of synthetic data should match the real data

This triple-loss approach produces higher quality synthetic data than a standard GAN alone.

Example:

## How It Works

TableGAN uses a DCGAN architecture adapted for tabular data with three loss components:

- **Adversarial loss**: Standard GAN objective for realism
- **Classification loss**: Ensures generated data preserves label column relationships
- **Information loss**: Minimizes statistical divergence between real and synthetic data

Reference: "Data Synthesis based on Generative Adversarial Networks" (Park et al., 2018)

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the training batch size. |
| `ClassificationWeight` | Gets or sets the weight of the classification loss. |
| `ClassifierDimensions` | Gets or sets the hidden layer sizes for the classifier head. |
| `DiscriminatorDimensions` | Gets or sets the hidden layer sizes for the discriminator. |
| `DiscriminatorDropout` | Gets or sets the dropout rate for discriminator hidden layers. |
| `DiscriminatorSteps` | Gets or sets the number of discriminator training steps per generator step. |
| `EmbeddingDimension` | Gets or sets the dimension of the random noise vector for the generator. |
| `Epochs` | Gets or sets the number of training epochs. |
| `GeneratorDimensions` | Gets or sets the hidden layer sizes for the generator. |
| `GradientPenaltyWeight` | Gets or sets the weight for the WGAN-GP gradient penalty term. |
| `InformationWeight` | Gets or sets the weight of the information loss (statistical similarity). |
| `LabelColumnIndex` | Gets or sets the index of the label column (for classification loss). |
| `LearningRate` | Gets or sets the learning rate. |
| `TrainClassifier` | Gets or sets whether the generator's loss includes the classification auxiliary term. |
| `VGMModes` | Gets or sets the number of VGM modes for continuous column transformation. |

