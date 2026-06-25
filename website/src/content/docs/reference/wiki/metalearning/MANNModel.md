---
title: "MANNModel<T, TInput, TOutput>"
description: "MANN model for inference with external memory."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

MANN model for inference with external memory.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MANNModel(IFullModel<,,>,ExternalMemory<>,MANNOptions<,,>,INumericOperations<>)` | Initializes a new instance of the MANNModel. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Metadata` | Gets the model metadata. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetModelMetadata` | Gets model metadata. |
| `GetParameters` | Gets controller parameters. |
| `Predict()` | Makes predictions using memory-augmented classification. |
| `Train(,)` | Training is not supported for adapted models. |
| `UpdateParameters(Vector<>)` | Parameter updates are not supported for adapted models. |

