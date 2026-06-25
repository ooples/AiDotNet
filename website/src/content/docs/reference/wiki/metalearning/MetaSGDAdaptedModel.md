---
title: "MetaSGDAdaptedModel<T, TInput, TOutput>"
description: "Wrapper model for Meta-SGD adapted models that includes the per-parameter optimizer."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Wrapper model for Meta-SGD adapted models that includes the per-parameter optimizer.

## How It Works

This model wraps an adapted model along with its per-parameter optimizer,
allowing for further adaptation or inspection of learned coefficients.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MetaSGDAdaptedModel(IFullModel<,,>,PerParameterOptimizer<,,>,MetaSGDOptions<,,>)` | Initializes a new instance of the MetaSGDAdaptedModel. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Metadata` | Gets the model metadata. |
| `Optimizer` | Gets the per-parameter optimizer (for inspection or further adaptation). |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetModelMetadata` | Gets the model metadata. |
| `GetParameters` | Gets the current model parameters. |
| `Predict()` | Makes predictions using the adapted model. |
| `SetParameters(Vector<>)` | Sets the model parameters. |
| `Train(,)` | Trains the model on the given data. |

