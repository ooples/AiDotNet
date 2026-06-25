---
title: "MAE<T>"
description: "MAE: Masked Autoencoder for Self-Supervised Vision Learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SelfSupervisedLearning`

MAE: Masked Autoencoder for Self-Supervised Vision Learning.

## For Beginners

MAE is a simple yet powerful self-supervised method.
It randomly masks a large portion (75%) of image patches, encodes only the visible
patches, and trains a decoder to reconstruct the original pixels of the masked patches.

## How It Works

**Key innovations:**

**Architecture:**

**Reference:** He et al., "Masked Autoencoders Are Scalable Vision Learners"
(CVPR 2022)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MAE(INeuralNetwork<>,INeuralNetwork<>,Int32,Int32,Double,SSLConfig)` | Initializes a new instance of the MAE class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Category` |  |
| `MaskRatio` | Gets the mask ratio (proportion of patches masked). |
| `Name` |  |
| `PatchSize` | Gets the patch size. |
| `RequiresMemoryBank` |  |
| `UsesMomentumEncoder` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Create(INeuralNetwork<>,INeuralNetwork<>,Int32,Int32,Double)` | Creates an MAE instance with default configuration. |
| `RouteGradientsToEncoder(Tensor<>,Int32[][],Tensor<>)` | Routes gradients from decoder output back to encoder through visible patch positions. |
| `RouteGradientsToEncoderDirect(Tensor<>,Int32[][],Tensor<>)` | Routes gradients directly to encoder when no decoder is present. |

