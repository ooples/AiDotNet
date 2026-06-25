---
title: "BarlowTwinsLoss<T>"
description: "Barlow Twins Loss - redundancy reduction loss for self-supervised learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SelfSupervisedLearning.Losses`

Barlow Twins Loss - redundancy reduction loss for self-supervised learning.

## For Beginners

Barlow Twins loss encourages the cross-correlation matrix
between embeddings of two augmented views to be close to the identity matrix.
This avoids both collapse and redundant features.

## How It Works

**Key insight:** Instead of contrastive learning, Barlow Twins achieves
representation quality through decorrelation - making different feature dimensions
capture different information.

**Loss components:**

**Loss formula:**

where C is the cross-correlation matrix and λ is the redundancy weight.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BarlowTwinsLoss(Double,Boolean)` | Initializes a new instance of the BarlowTwinsLoss class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Lambda` | Gets the lambda (redundancy reduction weight) parameter. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BatchNormalize(Tensor<>)` | Normalizes tensor along batch dimension (mean=0, std=1 for each feature). |
| `ComputeCrossCorrelation(Tensor<>,Tensor<>,Int32)` | Computes the cross-correlation matrix between two sets of embeddings. |
| `ComputeLoss(Tensor<>,Tensor<>)` | Computes the Barlow Twins loss between two views. |
| `ComputeLossWithGradients(Tensor<>,Tensor<>)` | Computes the Barlow Twins loss with gradients for backpropagation. |
| `OffDiagonalSum(Tensor<>)` | Computes the off-diagonal sum of a matrix (useful for monitoring). |

