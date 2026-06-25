---
title: "MaskDiceLoss<T>"
description: "Dice loss for mask prediction."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Losses`

Dice loss for mask prediction.

## For Beginners

Dice loss directly optimizes the Dice coefficient (F1 score),
making it better for imbalanced masks where foreground is much smaller than background.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MaskDiceLoss(Double)` | Creates a new mask Dice loss. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute(Tensor<>,Tensor<>)` | Computes Dice loss between predicted and target masks. |

