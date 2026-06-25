---
title: "FinGPT<T>"
description: "FinGPT neural network model for domain-specific financial language generation and analysis."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.NLP`

FinGPT neural network model for domain-specific financial language generation and analysis.

## For Beginners

FinGPT is an open-source financial language model that can
analyze news, generate financial reports, and perform sentiment analysis. Unlike
proprietary models like BloombergGPT, FinGPT is designed to be accessible and
fine-tunable by anyone. It can be adapted to specific financial tasks like stock
prediction from news headlines or risk assessment from annual reports.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FinGPT(NeuralNetworkArchitecture<>,FinGPTOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a FinGPT network in native mode for training. |
| `FinGPT(NeuralNetworkArchitecture<>,String,FinGPTOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a FinGPT network using a pretrained ONNX model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Executes CreateNewInstance for the FinGPT. |
| `DeserializeModelSpecificData(BinaryReader)` | Executes DeserializeModelSpecificData for the FinGPT. |
| `ExtractLayerReferences` | Executes ExtractLayerReferences for the FinGPT. |
| `ForecastNative(Tensor<>,Double[])` | Executes ForecastNative for the FinGPT. |
| `GetOptions` |  |
| `InitializeLayers` | Executes InitializeLayers for the FinGPT. |
| `SerializeModelSpecificData(BinaryWriter)` | Executes SerializeModelSpecificData for the FinGPT. |
| `TrainCore(Tensor<>,Tensor<>,Tensor<>)` | Executes TrainCore for the FinGPT. |
| `UpdateParameters(Vector<>)` | Executes UpdateParameters for the FinGPT. |
| `ValidateInputShape(Tensor<>)` | Executes ValidateInputShape for the FinGPT. |

