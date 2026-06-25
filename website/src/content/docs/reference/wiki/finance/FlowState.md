---
title: "FlowState<T>"
description: "FlowState — IBM's SSM-based Time Series Foundation Model (9.1M parameters)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Forecasting.Foundation`

FlowState — IBM's SSM-based Time Series Foundation Model (9.1M parameters).

## For Beginners

FlowState is a compact forecasting model from IBM that punches
well above its weight. With only 9.1 million parameters (tiny by modern standards), it
outperforms models 20 times its size. It uses state-space models, which process data
like a conveyor belt rather than looking at everything at once, making it very efficient
with long sequences of data like years of daily stock prices.

## How It Works

FlowState is IBM's State-Space Model based time series foundation model. Despite having
only 9.1M parameters (smallest in GIFT-Eval top 10), it outperforms models 20x its size
and generalizes to unseen timescales. It uses structured state spaces for linear-time
processing of long sequences.

**Reference:** IBM Research, "SSM Time Series Model", 2025.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FlowState(NeuralNetworkArchitecture<>,FlowStateOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a FlowState model in native mode. |
| `FlowState(NeuralNetworkArchitecture<>,String,FlowStateOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a FlowState model using a pretrained ONNX model. |

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
| `DenormalizeForecast(Tensor<>)` | RevIN reverse step (Kim et al. |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Evaluate(Tensor<>,Tensor<>)` |  |
| `Forecast(Tensor<>,Double[])` |  |
| `ForwardNativeForTraining(Tensor<>)` | Training-mode forward. |
| `GetFinancialMetrics` |  |
| `GetModelMetadata` |  |
| `GetNamedLayerActivations(Tensor<>)` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `PredictCore(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

