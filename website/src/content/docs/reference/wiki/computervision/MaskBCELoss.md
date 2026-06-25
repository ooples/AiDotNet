---
title: "MaskBCELoss<T>"
description: "Binary Cross-Entropy loss for mask prediction."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Losses`

Binary Cross-Entropy loss for mask prediction.

## For Beginners

BCE loss is the standard loss for binary segmentation masks.
It computes the cross-entropy between predicted probabilities and binary ground truth.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MaskBCELoss(Double)` | Creates a new mask BCE loss. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute(Tensor<>,Tensor<>)` | Computes BCE loss between predicted and target masks. |

