---
title: "CombinedMaskLoss<T>"
description: "Combined mask loss using BCE + Dice."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Losses`

Combined mask loss using BCE + Dice.

## For Beginners

Combining BCE and Dice loss often gives better results
than using either alone. BCE provides pixel-level supervision while Dice optimizes
the overall mask quality.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CombinedMaskLoss(Double,Double)` | Creates a combined BCE + Dice loss. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute(Tensor<>,Tensor<>)` | Computes combined loss. |

