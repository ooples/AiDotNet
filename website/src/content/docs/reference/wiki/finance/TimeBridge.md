---
title: "TimeBridge<T>"
description: "TimeBridge — Non-Stationarity Matters for Time Series Foundation Models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Forecasting.Foundation`

TimeBridge — Non-Stationarity Matters for Time Series Foundation Models.

## For Beginners

Most forecasting models normalize data (remove trends and
scale to a standard range) before processing, which can lose important information about
long-term trends and sudden shifts. TimeBridge solves this by "bridging" the gap: it
saves the non-stationary information that normalization removes and adds it back to the
predictions, giving you forecasts that correctly capture upward trends, seasonal shifts,
and level changes.

## How It Works

TimeBridge addresses the critical non-stationarity gap in time series foundation models.
It introduces a bridge mechanism that preserves and restores non-stationary information
(trends, level shifts) that is typically lost during standard normalization.

**Reference:** "TimeBridge: Non-Stationarity Matters for Long-term Time Series Forecasting", 2024.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TimeBridge(NeuralNetworkArchitecture<>,String,TimeBridgeOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a TimeBridge model using a pretrained ONNX model. |
| `TimeBridge(NeuralNetworkArchitecture<>,TimeBridgeOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a TimeBridge model in native mode for training or fine-tuning. |

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

