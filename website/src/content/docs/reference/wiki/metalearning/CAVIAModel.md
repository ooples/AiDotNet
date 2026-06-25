---
title: "CAVIAModel<T, TInput, TOutput>"
description: "CAVIA inference model that uses adapted context parameters for predictions."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

CAVIA inference model that uses adapted context parameters for predictions.

## For Beginners

After adapting CAVIA to a new task, you get this model.
It automatically adds the learned task context to your inputs, so you can use it
just like any other model: call Predict() with new examples.

## How It Works

This model wraps the meta-learned body parameters with task-specific context parameters.
It is returned by `IMetaLearningTask{` and provides
fast inference by augmenting inputs with the adapted context before passing to the model.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CAVIAModel(IFullModel<,,>,Vector<>,CAVIAOptions<,,>,INumericOperations<>)` | Initializes a new instance of the CAVIAModel with adapted context. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdaptedContext` | Gets the adapted context vector for inspection or further processing. |
| `Metadata` | Gets the model metadata. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AugmentInput(,Vector<>)` | Augments input by injecting context according to the configured injection mode. |
| `GetModelMetadata` | Gets metadata about this CAVIA inference model. |
| `GetParameters` | Gets model parameters (not applicable for CAVIA inference models). |
| `Predict()` | Makes predictions by augmenting input with the adapted context and running through the body model. |
| `Train(,)` | Trains the model (not applicable for CAVIA inference models). |
| `UpdateParameters(Vector<>)` | Updates model parameters (not applicable for CAVIA inference models). |

