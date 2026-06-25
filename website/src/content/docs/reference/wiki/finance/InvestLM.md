---
title: "InvestLM<T>"
description: "InvestLM neural network model specialized for investment professionals and research."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.NLP`

InvestLM neural network model specialized for investment professionals and research.

## For Beginners

InvestLM is a language model fine-tuned specifically for
investment professionals. It can analyze investment opportunities, summarize financial
reports, provide market commentary, and answer questions about portfolio strategies.
Think of it as an AI research analyst that understands investment terminology, valuation
methods, and market dynamics.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `InvestLM(NeuralNetworkArchitecture<>,InvestLMOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates an InvestLM network in native mode for training. |
| `InvestLM(NeuralNetworkArchitecture<>,String,InvestLMOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates an InvestLM network using a pretrained ONNX model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Executes CreateNewInstance for the InvestLM. |
| `DeserializeModelSpecificData(BinaryReader)` | Executes DeserializeModelSpecificData for the InvestLM. |
| `ExtractLayerReferences` | Executes ExtractLayerReferences for the InvestLM. |
| `ForecastNative(Tensor<>,Double[])` | Executes ForecastNative for the InvestLM. |
| `GetOptions` |  |
| `InitializeLayers` | Executes InitializeLayers for the InvestLM. |
| `SerializeModelSpecificData(BinaryWriter)` | Executes SerializeModelSpecificData for the InvestLM. |
| `TrainCore(Tensor<>,Tensor<>,Tensor<>)` | Executes TrainCore for the InvestLM. |
| `UpdateParameters(Vector<>)` | Executes UpdateParameters for the InvestLM. |
| `ValidateInputShape(Tensor<>)` | Executes ValidateInputShape for the InvestLM. |

