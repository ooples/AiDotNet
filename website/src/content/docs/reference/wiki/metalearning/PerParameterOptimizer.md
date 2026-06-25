---
title: "PerParameterOptimizer<T, TInput, TOutput>"
description: "Per-parameter optimizer for Meta-SGD that learns individual optimization coefficients."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Per-parameter optimizer for Meta-SGD that learns individual optimization coefficients.

## For Beginners

This is a special optimizer where each weight in the
network gets its own set of optimization settings that are learned during
meta-training.

## How It Works

This optimizer maintains learned coefficients for each parameter:

- Learning rates: α_i for each parameter
- Momentum: β_i for each parameter (optional)
- Direction: d_i for each parameter (optional)
- Adam parameters: beta1, beta2, epsilon (if using Adam)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PerParameterOptimizer(Int32,MetaSGDOptions<,,>,IEngine)` | Initializes a new instance of the PerParameterOptimizer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumParameters` | Gets the number of model parameters this optimizer manages. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyRegularization` | Applies regularization and clipping to learned coefficients. |
| `Clone` | Creates a deep copy of this per-parameter optimizer. |
| `GetLearningRate(Int32)` | Gets the learning rate for a specific parameter. |
| `GetMetaParameterCount` | Gets the total number of meta-parameters being learned. |
| `SetAdamBeta1(Int32,)` | Sets Adam beta1 for a specific parameter. |
| `SetAdamBeta2(Int32,)` | Sets Adam beta2 for a specific parameter. |
| `SetAdamEpsilon(Int32,)` | Sets Adam epsilon for a specific parameter. |
| `SetDirection(Int32,)` | Sets the direction for a specific parameter. |
| `SetLearningRate(Int32,)` | Sets the learning rate for a specific parameter. |
| `SetMomentum(Int32,)` | Sets the momentum for a specific parameter. |
| `UpdateMetaParameters(Vector<>)` | Updates the meta-parameters (learned coefficients) of the optimizer. |
| `UpdateParameter(Int32,,)` | Updates a single parameter using its learned optimization coefficients. |

