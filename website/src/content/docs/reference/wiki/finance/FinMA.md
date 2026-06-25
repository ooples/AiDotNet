---
title: "FinMA<T>"
description: "FinMA (Financial Multi-Agent) neural network model for collaborative financial task solving."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.NLP`

FinMA (Financial Multi-Agent) neural network model for collaborative financial task solving.

## For Beginners

FinMA is a financial language model from the PIXIU project
that can handle a wide range of financial tasks including sentiment analysis, named
entity recognition in SEC filings, stock movement prediction, and financial question
answering. It was instruction-tuned on 136K financial task examples, making it a
versatile financial AI assistant.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FinMA(NeuralNetworkArchitecture<>,FinMAOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a FinMA network in native mode for training. |
| `FinMA(NeuralNetworkArchitecture<>,String,FinMAOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a FinMA network using a pretrained ONNX model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Executes CreateNewInstance for the FinMA. |
| `DeserializeModelSpecificData(BinaryReader)` | Executes DeserializeModelSpecificData for the FinMA. |
| `ExtractLayerReferences` | Executes ExtractLayerReferences for the FinMA. |
| `ForecastNative(Tensor<>,Double[])` | Executes ForecastNative for the FinMA. |
| `GetOptions` |  |
| `InitializeLayers` | Executes InitializeLayers for the FinMA. |
| `SerializeModelSpecificData(BinaryWriter)` | Executes SerializeModelSpecificData for the FinMA. |
| `TrainCore(Tensor<>,Tensor<>,Tensor<>)` | Executes TrainCore for the FinMA. |
| `UpdateParameters(Vector<>)` | Executes UpdateParameters for the FinMA. |
| `ValidateInputShape(Tensor<>)` | Executes ValidateInputShape for the FinMA. |

