---
title: "GPT4TS<T>"
description: "GPT4TS — One Fits All: Power General Time Series Analysis by Pretrained LM."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Forecasting.Foundation`

GPT4TS — One Fits All: Power General Time Series Analysis by Pretrained LM.

## For Beginners

GPT4TS takes a language model (GPT-2) that was trained to
predict text and repurposes it for time series forecasting. The core idea is that patterns
in sequences of numbers are similar to patterns in sequences of words. The language model
stays frozen (unchanged) while small task-specific layers are added on top, making this
approach surprisingly effective with minimal training.

## How It Works

GPT4TS uses a frozen GPT-2 backbone with task-specific heads for time series forecasting,
classification, and anomaly detection. It demonstrates that pretrained language models
transfer effectively to time series tasks without fine-tuning the backbone.

**Reference:** Zhou et al., "One Fits All: Power General Time Series Analysis by Pretrained LM", 2023.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GPT4TS(NeuralNetworkArchitecture<>,GPT4TSOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a GPT4TS model in native mode for training or fine-tuning. |
| `GPT4TS(NeuralNetworkArchitecture<>,String,GPT4TSOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a GPT4TS model using a pretrained ONNX model. |

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

