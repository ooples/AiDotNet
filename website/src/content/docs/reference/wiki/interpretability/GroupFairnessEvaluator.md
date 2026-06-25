---
title: "GroupFairnessEvaluator<T>"
description: "Group-level fairness evaluator that focuses on equalized performance across groups."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability`

Group-level fairness evaluator that focuses on equalized performance across groups.
Computes equal opportunity and equalized odds when actual labels are available.
Focuses on ensuring similar error rates across demographic groups.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GroupFairnessEvaluator` | Initializes a new instance of the GroupFairnessEvaluator class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetFairnessMetrics(IFullModel<,Matrix<>,Vector<>>,Matrix<>,Int32,Vector<>)` | Computes group-level fairness metrics focusing on performance equity. |

