---
title: "BYOLLoss<T>"
description: "BYOL (Bootstrap Your Own Latent) Loss - a simple cosine similarity loss."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SelfSupervisedLearning.Losses`

BYOL (Bootstrap Your Own Latent) Loss - a simple cosine similarity loss.

## For Beginners

BYOL uses a simple Mean Squared Error (MSE) loss between
normalized predictions and normalized targets. Unlike contrastive methods, it doesn't
require negative samples.

## How It Works

**Key insight:** BYOL avoids collapse through asymmetric architecture
(predictor on one branch) and momentum updates, not through negative samples.

**Loss formula:**

where p is prediction, z' is target projection, and sg() means stop-gradient.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BYOLLoss(Boolean,Boolean)` | Initializes a new instance of the BYOLLoss class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeLoss(Tensor<>,Tensor<>)` | Computes the BYOL loss between online predictions and target projections. |
| `ComputeLossWithGradients(Tensor<>,Tensor<>)` | Computes BYOL loss with gradients for backpropagation. |
| `ComputeMSELoss(Tensor<>,Tensor<>)` | Computes the mean squared error between normalized embeddings. |
| `ComputeSymmetricLoss(Tensor<>,Tensor<>,Tensor<>,Tensor<>)` | Computes symmetric BYOL loss for both views. |

