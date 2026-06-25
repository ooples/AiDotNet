---
title: "ContributionMethod"
description: "Specifies the method used to evaluate client contributions in federated learning."
section: "API Reference"
---

`Enums` · `AiDotNet.Models.Options`

Specifies the method used to evaluate client contributions in federated learning.

## For Beginners

Not all clients contribute equally to the global model. Some provide
high-quality data that greatly improves accuracy, while others may free-ride. These methods
measure each client's contribution so you can fairly compensate them or detect problems.

## Fields

| Field | Summary |
|:-----|:--------|
| `DataShapley` | Data Shapley: Monte Carlo approximation of Shapley values. |
| `LightweightShapley` | Lightweight Shapley: efficient approximation using gradient similarity instead of full retraining. |
| `Prototypical` | Prototypical contribution: evaluates contribution using prototype representations. |
| `ShapleyValue` | Exact Shapley value: measures marginal contribution by testing all possible coalitions. |

