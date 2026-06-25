---
title: "OCTGANOptions<T>"
description: "Configuration options for OCT-GAN (One-Class Tabular GAN), designed for generating synthetic data with a focus on minority/imbalanced classes."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for OCT-GAN (One-Class Tabular GAN), designed for generating
synthetic data with a focus on minority/imbalanced classes.

## For Beginners

OCT-GAN is designed for imbalanced datasets (e.g., fraud detection
where 99% of transactions are normal and only 1% are fraud).

Instead of generating all data equally, it focuses on the minority class:

1. The discriminator learns what minority samples look like
2. The generator tries to create realistic minority samples
3. This produces better synthetic oversampling than random duplication

Example:

## How It Works

OCT-GAN addresses class imbalance by using a one-class discriminator that focuses
on the minority class characteristics, making it ideal for oversampling.

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the training batch size. |
| `CenterUpdateMomentum` | Gets or sets the momentum for exponential moving average updates of the SVDD center. |
| `DiscriminatorDimensions` | Gets or sets the hidden layer sizes for the discriminator. |
| `DiscriminatorDropout` | Gets or sets the dropout rate for discriminator hidden layers. |
| `DiscriminatorSteps` | Gets or sets the number of discriminator training steps per generator step. |
| `EmbeddingDimension` | Gets or sets the dimension of the noise vector. |
| `Epochs` | Gets or sets the number of training epochs. |
| `GeneratorDimensions` | Gets or sets the hidden layer sizes for the generator. |
| `GradientPenaltyWeight` | Gets or sets the weight for the WGAN-GP gradient penalty term. |
| `LabelColumnIndex` | Gets or sets the index of the label column. |
| `LearningRate` | Gets or sets the learning rate. |
| `MinorityClassValue` | Gets or sets the minority class value to focus on. |
| `SVDDEmbeddingDimension` | Gets or sets the dimension of the discriminator's embedding space for the SVDD objective. |
| `VGMModes` | Gets or sets the number of VGM modes for continuous column encoding. |

