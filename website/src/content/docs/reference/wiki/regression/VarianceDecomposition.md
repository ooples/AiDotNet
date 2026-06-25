---
title: "VarianceDecomposition<T>"
description: "Contains the full variance decomposition results from a mixed model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Regression.MixedEffects`

Contains the full variance decomposition results from a mixed model.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VarianceDecomposition` | Initializes a new variance decomposition. |

## Properties

| Property | Summary |
|:-----|:--------|
| `RandomEffectVariances` | Gets or sets the random effect variance components. |
| `ResidualVariance` | Gets or sets the residual (within-group) variance. |
| `TotalVariance` | Gets the total variance (sum of all components). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeICC(Int32)` | Computes the Intraclass Correlation Coefficient (ICC) for a specific random effect. |
| `GetVarianceProportions` | Gets variance proportions for all components. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Numeric operations for type T. |

