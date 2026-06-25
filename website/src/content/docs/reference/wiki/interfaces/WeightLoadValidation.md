---
title: "WeightLoadValidation"
description: "Result of weight validation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interfaces`

Result of weight validation.

## Properties

| Property | Summary |
|:-----|:--------|
| `IsComplete` | Whether all model parameters have matching weights. |
| `IsValid` | Whether validation passed (all matched weights have correct shapes). |
| `Matched` | Parameter names in the model that matched weights. |
| `MissingParameters` | Model parameters that have no corresponding weight. |
| `ShapeMismatches` | Weights where shape doesn't match the model parameter. |
| `UnmatchedWeights` | Weight names that could not be mapped to any model parameter. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ToString` |  |

