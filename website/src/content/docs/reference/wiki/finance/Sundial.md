---
title: "Sundial<T>"
description: "Sundial — A Family of Highly Capable Time Series Foundation Models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Forecasting.Foundation`

Sundial — A Family of Highly Capable Time Series Foundation Models.

## For Beginners

Sundial is a time series forecasting model that works like
GPT but for numbers instead of words. It groups data into patches (chunks) and predicts
future patches one at a time. Despite using fewer parameters than competing models, it
achieves better accuracy, making it a practical choice when you need strong forecasting
performance without enormous computational resources.

## How It Works

Sundial is a decoder-only time series foundation model that outperforms Time-MoE with
fewer parameters (4.71% average MSE reduction). It uses a GPT-style autoregressive
architecture with patch-based tokenization and quantile forecasting.

**Reference:** "Sundial: A Family of Highly Capable Time Series Foundation Models", 2025.
https://arxiv.org/abs/2502.00816

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Sundial(NeuralNetworkArchitecture<>,String,SundialOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a Sundial model using a pretrained ONNX model. |
| `Sundial(NeuralNetworkArchitecture<>,SundialOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a Sundial model in native mode for training or fine-tuning. |

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

