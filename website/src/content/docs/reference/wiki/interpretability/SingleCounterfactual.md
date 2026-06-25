---
title: "SingleCounterfactual<T>"
description: "Represents a single counterfactual explanation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Represents a single counterfactual explanation.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SingleCounterfactual(Vector<>,,List<FeatureChange<>>,Double)` | Initializes a new single counterfactual. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Changes` | Features that were changed. |
| `CounterfactualInstance` | The counterfactual instance. |
| `Distance` | Normalized distance from original. |
| `Prediction` | Prediction for the counterfactual. |
| `Sparsity` | Number of features changed. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ToString` | Returns string representation. |

