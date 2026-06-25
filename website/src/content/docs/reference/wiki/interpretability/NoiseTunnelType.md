---
title: "NoiseTunnelType"
description: "Types of noise tunnel aggregation methods."
section: "API Reference"
---

`Enums` · `AiDotNet.Interpretability.Explainers`

Types of noise tunnel aggregation methods.

## For Beginners

These methods determine how multiple noisy attributions
are combined into a single final attribution.

## Fields

| Field | Summary |
|:-----|:--------|
| `SmoothGrad` | SmoothGrad: Simple average of attributions. |
| `SquaredSmoothGrad` | SmoothGrad-Squared: Average of squared attributions, then sqrt. |
| `VarGrad` | VarGrad: Variance of attributions. |

