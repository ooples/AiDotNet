---
title: "MaskFocalLoss<T>"
description: "Focal loss for mask prediction (addresses class imbalance)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Losses`

Focal loss for mask prediction (addresses class imbalance).

## For Beginners

Focal loss down-weights easy examples (confident predictions)
and focuses training on hard examples. This helps with the severe class imbalance
in instance segmentation where background dominates.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MaskFocalLoss(Double,Double,Double)` | Creates a new mask focal loss. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute(Tensor<>,Tensor<>)` | Computes focal loss between predicted and target masks. |

