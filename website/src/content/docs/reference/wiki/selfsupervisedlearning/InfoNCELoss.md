---
title: "InfoNCELoss<T>"
description: "InfoNCE (Noise Contrastive Estimation) Loss for contrastive learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SelfSupervisedLearning.Losses`

InfoNCE (Noise Contrastive Estimation) Loss for contrastive learning.

## For Beginners

InfoNCE is the loss function used in MoCo (Momentum Contrast).
It's similar to NT-Xent but designed to work efficiently with a large memory queue of
negative samples.

## How It Works

**Key differences from NT-Xent:**

**Loss formula:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `InfoNCELoss(Double,Boolean)` | Initializes a new instance of the InfoNCELoss class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Temperature` | Gets the temperature parameter. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AiDotNet#Interfaces#IContrastiveLoss{T}#ComputeLoss(Tensor<>,Tensor<>)` | IContrastiveLoss implementation — delegates to in-batch contrastive loss. |
| `ComputeAccuracy(Tensor<>,Tensor<>,Tensor<>)` | Computes accuracy of the contrastive task (useful for monitoring). |
| `ComputeLoss(Tensor<>,Tensor<>,Tensor<>)` | Computes the InfoNCE loss using queries, positive keys, and negative keys from memory bank. |
| `ComputeLossInBatch(Tensor<>,Tensor<>)` | Computes InfoNCE loss with in-batch negatives (SimCLR style but asymmetric). |
| `ComputeLossInBatchWithGradients(Tensor<>,Tensor<>)` | Computes InfoNCE loss with in-batch negatives and gradients. |
| `ComputeLossWithGradients(Tensor<>,Tensor<>,Tensor<>)` | Computes InfoNCE loss with gradients. |

