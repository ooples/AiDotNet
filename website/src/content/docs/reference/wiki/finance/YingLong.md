---
title: "YingLong<T>"
description: "YingLong — Alibaba's Enterprise Time Series Foundation Model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Forecasting.Foundation`

YingLong — Alibaba's Enterprise Time Series Foundation Model.

## For Beginners

YingLong is Alibaba's general-purpose time series forecasting
model, pre-trained on massive amounts of data from Alibaba's cloud infrastructure. It is
designed to handle enterprise-scale workloads like demand forecasting, capacity planning,
and resource allocation. Think of it as a forecasting model that has already learned from
one of the largest e-commerce and cloud platforms in the world.

## How It Works

YingLong is Alibaba's transformer-based time series foundation model designed for
general-purpose forecasting with a focus on cloud and enterprise workloads.
Pre-trained on large-scale data from Alibaba's data infrastructure.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `YingLong(NeuralNetworkArchitecture<>,String,YingLongOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a YingLong model using a pretrained ONNX model. |
| `YingLong(NeuralNetworkArchitecture<>,YingLongOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a YingLong model in native mode for training or fine-tuning. |

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

