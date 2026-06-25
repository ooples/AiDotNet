---
title: "MetaSGDUpdateRuleType"
description: "Update rule types for Meta-SGD per-parameter optimization."
section: "API Reference"
---

`Enums` · `AiDotNet.MetaLearning.Options`

Update rule types for Meta-SGD per-parameter optimization.

## How It Works

These define the base optimization algorithm that Meta-SGD learns to configure
on a per-parameter basis.

## Fields

| Field | Summary |
|:-----|:--------|
| `AdaDelta` | AdaDelta optimizer with learned decay per parameter. |
| `AdaGrad` | AdaGrad optimizer with learned accumulation per parameter. |
| `Adam` | Adam optimizer with optionally learned beta parameters per parameter. |
| `RMSprop` | RMSprop optimizer with learned decay rates per parameter. |
| `SGD` | Standard Stochastic Gradient Descent with learned per-parameter learning rates. |
| `SGDWithMomentum` | SGD with learned per-parameter momentum coefficients. |

