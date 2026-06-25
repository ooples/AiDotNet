---
title: "OptimizationMethod<T>"
description: "The optimization method to use."
section: "API Reference"
---

`Enums` · `AiDotNet.GaussianProcesses`

The optimization method to use.

## Fields

| Field | Summary |
|:-----|:--------|
| `BayesianOptimization` | Bayesian optimization using GP surrogate. |
| `GradientDescent` | Gradient-based optimization (requires differentiable kernel). |
| `GridSearch` | Grid search over specified parameter values. |
| `RandomSearch` | Random search over parameter ranges. |

