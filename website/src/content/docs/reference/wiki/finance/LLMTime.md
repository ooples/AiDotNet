---
title: "LLMTime<T>"
description: "LLM-Time — Zero-Shot Time Series Forecasting via LLM Tokenization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Forecasting.Foundation`

LLM-Time — Zero-Shot Time Series Forecasting via LLM Tokenization.

## For Beginners

LLM-Time takes an unconventional approach: it converts your
numerical data into text (like "3.2, 3.5, 3.8, ...") and asks a large language model to
predict the next numbers, just like predicting the next word in a sentence. Remarkably,
this works without any training on time series data at all. The LLM already understands
number patterns from its language training.

## How It Works

LLM-Time converts numeric time series into text strings and uses pretrained LLMs (GPT-3, LLaMA)
for zero-shot forecasting by treating the task as next-token prediction on numerical text.
No fine-tuning is required—the LLM backbone is frozen.

**Reference:** Gruver et al., "Large Language Models Are Zero-Shot Time Series Forecasters", NeurIPS 2023.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LLMTime(NeuralNetworkArchitecture<>,LLMTimeOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates an LLMTime model in native mode for training or fine-tuning. |
| `LLMTime(NeuralNetworkArchitecture<>,String,LLMTimeOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates an LLMTime model using a pretrained ONNX model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsChannelIndependent` |  |
| `MaxContextLength` |  |
| `MaxPredictionHorizon` |  |
| `ModelSize` |  |
| `NumFeatures` |  |
| `PatchSize` |  |
| `PredictionHorizon` |  |
| `SequenceLength` |  |
| `Stride` |  |
| `SupportsTraining` |  |
| `UseNativeMode` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyInstanceNormalization(Tensor<>)` |  |
| `AutoregressiveForecast(Tensor<>,Int32)` |  |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Evaluate(Tensor<>,Tensor<>)` |  |
| `Forecast(Tensor<>,Double[])` |  |
| `GetFinancialMetrics` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `PredictCore(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

