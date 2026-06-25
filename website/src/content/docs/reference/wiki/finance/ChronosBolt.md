---
title: "ChronosBolt<T>"
description: "Chronos-Bolt — Fast Non-Autoregressive Time Series Forecasting from the Chronos Family."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Forecasting.Foundation`

Chronos-Bolt — Fast Non-Autoregressive Time Series Forecasting from the Chronos Family.

## For Beginners

Chronos-Bolt is a faster version of the Chronos time series
forecasting model from Amazon. While the original Chronos generates predictions one step
at a time (like writing a sentence word by word), Chronos-Bolt outputs all predictions at
once (like writing the whole sentence in one go). This makes it much faster while still
providing probabilistic forecasts that tell you the range of likely future values.

## How It Works

Chronos-Bolt uses an encoder-decoder architecture with direct quantile forecasting
(non-autoregressive), making it significantly faster than autoregressive Chronos v1/v2.
The encoder processes the input context and the decoder directly outputs all forecast
quantiles in a single forward pass without iterative generation.

**Reference:** Part of Amazon Chronos family, Nov 2024.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ChronosBolt(NeuralNetworkArchitecture<>,ChronosBoltOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a Chronos-Bolt model in native mode. |
| `ChronosBolt(NeuralNetworkArchitecture<>,String,ChronosBoltOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a Chronos-Bolt model using a pretrained ONNX model. |

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
| `ForwardForTraining(Tensor<>)` | Tape-aware training forward. |
| `GetFinancialMetrics` |  |
| `GetModelMetadata` |  |
| `GetNamedLayerActivations(Tensor<>)` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `PredictCore(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `SetTrainingMode(Boolean)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

