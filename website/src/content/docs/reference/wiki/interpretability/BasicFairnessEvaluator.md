---
title: "BasicFairnessEvaluator<T>"
description: "Basic fairness evaluator that computes only fundamental fairness metrics."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability`

Basic fairness evaluator that computes only fundamental fairness metrics.
Includes demographic parity (statistical parity difference) and disparate impact.
Does not require actual labels.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BasicFairnessEvaluator` | Initializes a new instance of the BasicFairnessEvaluator class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetFairnessMetrics(IFullModel<,Matrix<>,Vector<>>,Matrix<>,Int32,Vector<>)` | Computes basic fairness metrics (demographic parity and disparate impact). |

