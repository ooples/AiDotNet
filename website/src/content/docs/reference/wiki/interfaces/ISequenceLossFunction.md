---
title: "ISequenceLossFunction<T>"
description: "Interface for sequence loss functions that operate on variable-length sequences."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for sequence loss functions that operate on variable-length sequences.

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateGradient(Tensor<>,Int32[][],Int32[],Int32[])` | Calculates the gradient of the loss with respect to the inputs. |
| `CalculateLoss(Tensor<>,Int32[][],Int32[],Int32[])` | Calculates the loss for sequence data. |

