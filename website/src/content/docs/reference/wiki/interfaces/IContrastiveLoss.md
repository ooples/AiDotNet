---
title: "IContrastiveLoss<T>"
description: "Interface for contrastive and self-supervised loss functions that operate on pairs of embeddings/representations rather than predictions vs ground truth labels."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for contrastive and self-supervised loss functions that operate on pairs
of embeddings/representations rather than predictions vs ground truth labels.

## How It Works

Unlike `ILossFunction` which compares predictions to actual labels,
contrastive losses compare two views/augmentations of the same data to learn
representations. Examples include SimCLR's NT-Xent, BYOL's regression loss,
and Barlow Twins' redundancy reduction loss.

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeLoss(Tensor<>,Tensor<>)` | Computes the contrastive loss between two embedding tensors. |

