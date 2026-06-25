---
title: "SAINT<T>"
description: "SAINT (Self-Attention and Intersample Attention Transformer) for tabular data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Risk`

SAINT (Self-Attention and Intersample Attention Transformer) for tabular data.

## For Beginners

SAINT improves on standard transformers for tabular data by paying attention
not just to the relationships between columns (features) but also between rows (samples).
This allows it to learn from similar examples in the batch, much like a nearest-neighbor approach
but fully learnable.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SAINT` | Initializes a new instance of the SAINT model. |
| `SAINT(NeuralNetworkArchitecture<>,String,SAINTOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new instance from ONNX. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AdjustForRisk(Tensor<>,)` | Adjusts action for risk. |
| `CalculateRisk(Tensor<>)` | Calculates risk using SAINT. |
| `CreateNewInstance` | Creates a new instance of the SAINT model with the same configuration. |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the neural network layers for SAINT. |
| `UpdateParameters(Vector<>)` | Updates the model parameters from a flat parameter vector. |

