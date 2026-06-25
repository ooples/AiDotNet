---
title: "ComputeCost"
description: "Relative computational cost of an operation."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Relative computational cost of an operation.

## Fields

| Field | Summary |
|:-----|:--------|
| `High` | Multiple transcendentals or reductions (GELU, Swish, Mish, Softmax). |
| `Low` | Simple comparison/max (ReLU, LeakyReLU, ThresholdedReLU). |
| `Medium` | One transcendental function (Sigmoid, Tanh, ELU). |

