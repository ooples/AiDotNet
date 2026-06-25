---
title: "TabTransformer<T>"
description: "TabTransformer model for tabular data, using transformers for categorical features."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Risk`

TabTransformer model for tabular data, using transformers for categorical features.

## For Beginners

TabTransformer applies the power of Transformers (like BERT/GPT) specifically
to the categorical parts of your data (e.g., "Sector", "Country", "Rating"). It learns deep
contextual embeddings for these categories before combining them with numerical data.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TabTransformer` | Initializes a new instance of the TabTransformer model. |
| `TabTransformer(NeuralNetworkArchitecture<>,String,TabTransformerOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new instance from ONNX. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AdjustForRisk(Tensor<>,)` | Adjusts action for risk. |
| `CalculateRisk(Tensor<>)` | Calculates risk using TabTransformer. |
| `CreateNewInstance` | Creates a new instance of the TabTransformer model with the same configuration. |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the neural network layers for TabTransformer. |
| `UpdateParameters(Vector<>)` | Updates the model parameters from a flat parameter vector. |

