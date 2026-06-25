---
title: "OptimizationMode"
description: "Specifies the mode of optimization for an optimizer."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Specifies the mode of optimization for an optimizer.

## For Beginners

Think of this as choosing what the optimizer is allowed to change. It can select
which features (input variables) to use, adjust the model's internal parameters, or do both. This gives you
control over how the optimizer improves your model.

## How It Works

OptimizationMode determines what aspects of a model the optimizer will modify during the optimization process.
This can include feature selection (choosing which features to use), parameter adjustment (modifying model parameters),
or both.

## Fields

| Field | Summary |
|:-----|:--------|
| `Both` | Optimize both feature selection and model parameters. |
| `FeatureSelectionOnly` | Optimize only feature selection (which features to include in the model). |
| `ParametersOnly` | Optimize only model parameters (adjust existing model parameters). |

