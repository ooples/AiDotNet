---
title: "TabNet<T>"
description: "TabNet model for tabular data learning, combining tree-based interpretability with deep learning performance."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Risk`

TabNet model for tabular data learning, combining tree-based interpretability with deep learning performance.

## For Beginners

TabNet is designed specifically for data that comes in tables (rows and columns),
like spreadsheets or databases. Traditional neural networks struggle with this kind of data compared to
decision trees (like XGBoost). TabNet tries to get the best of both worlds: the accuracy of trees
and the end-to-end learning of neural networks.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TabNet` | Initializes a new instance with default architecture settings. |
| `TabNet(NeuralNetworkArchitecture<>,String,TabNetOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new instance of the TabNet model from a pretrained ONNX model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AdjustForRisk(Tensor<>,)` | Adjusts action for risk. |
| `CalculateRisk(Tensor<>)` | Calculates risk using the TabNet model. |
| `CreateNewInstance` | Creates a new instance of the TabNet model with the same configuration. |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the neural network layers for TabNet. |
| `UpdateParameters(Vector<>)` | Updates the model parameters from a flat parameter vector. |

## Fields

| Field | Summary |
|:-----|:--------|
| `DefaultInputSize` | Initializes a new instance of the TabNet model. |

