---
title: "BloombergGPT<T>"
description: "BloombergGPT neural network model for comprehensive financial language processing."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.NLP`

BloombergGPT neural network model for comprehensive financial language processing.

## For Beginners

BloombergGPT is a 50-billion parameter language model
trained on a mix of general text and Bloomberg's massive financial data archive. Unlike
general-purpose LLMs, it deeply understands financial jargon, SEC filings, earnings
reports, and market commentary. It can analyze financial documents, answer questions
about markets, and generate financial text with domain-specific accuracy.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BloombergGPT(NeuralNetworkArchitecture<>,BloombergGPTOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a BloombergGPT network in native mode for training. |
| `BloombergGPT(NeuralNetworkArchitecture<>,String,BloombergGPTOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a BloombergGPT network using a pretrained ONNX model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Executes CreateNewInstance for the BloombergGPT. |
| `DeserializeModelSpecificData(BinaryReader)` | Executes DeserializeModelSpecificData for the BloombergGPT. |
| `ExtractLayerReferences` | Executes ExtractLayerReferences for the BloombergGPT. |
| `ForecastNative(Tensor<>,Double[])` | Executes ForecastNative for the BloombergGPT. |
| `GetOptions` |  |
| `InitializeLayers` | Executes InitializeLayers for the BloombergGPT. |
| `SerializeModelSpecificData(BinaryWriter)` | Executes SerializeModelSpecificData for the BloombergGPT. |
| `TrainCore(Tensor<>,Tensor<>,Tensor<>)` | Executes TrainCore for the BloombergGPT. |
| `UpdateParameters(Vector<>)` | Executes UpdateParameters for the BloombergGPT. |
| `ValidateInputShape(Tensor<>)` | Executes ValidateInputShape for the BloombergGPT. |

