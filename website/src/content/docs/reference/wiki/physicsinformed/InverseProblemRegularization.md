---
title: "InverseProblemRegularization"
description: "Specifies the type of regularization for inverse problems."
section: "API Reference"
---

`Enums` · `AiDotNet.PhysicsInformed.Interfaces`

Specifies the type of regularization for inverse problems.

## Fields

| Field | Summary |
|:-----|:--------|
| `Bayesian` | Bayesian regularization using prior distributions. |
| `ElasticNet` | Elastic Net: Combination of L1 and L2. |
| `L1Lasso` | L1 (Lasso) regularization: Prefers sparse parameters. |
| `L2Tikhonov` | L2 (Tikhonov) regularization: Prefers small parameter values. |
| `None` | No regularization (may be unstable). |
| `TotalVariation` | Total Variation regularization: Prefers smooth parameter fields. |

