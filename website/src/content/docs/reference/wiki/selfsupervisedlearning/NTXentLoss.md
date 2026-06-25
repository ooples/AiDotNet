---
title: "NTXentLoss<T>"
description: "Normalized Temperature-scaled Cross Entropy Loss (NT-Xent) for contrastive learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SelfSupervisedLearning.Losses`

Normalized Temperature-scaled Cross Entropy Loss (NT-Xent) for contrastive learning.

## For Beginners

NT-Xent is the loss function used in SimCLR. It encourages
representations of augmented views of the same image to be similar, while pushing apart
representations of different images.

## How It Works

**How it works:**

**Loss formula:**

Where τ (tau) is the temperature parameter.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NTXentLoss(Double,Boolean)` | Initializes a new instance of the NTXentLoss class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Temperature` | Gets the temperature parameter. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeLoss(Tensor<>,Tensor<>)` | Computes the NT-Xent loss for a batch of positive pairs. |
| `ComputeLossWithGradients(Tensor<>,Tensor<>)` | Computes the NT-Xent loss and returns gradients. |

