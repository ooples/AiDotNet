---
title: "Kairos<T>"
description: "Kairos — Adaptive and Generalizable Time Series Foundation Model with Mixture-of-Size Encoder."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Forecasting.Foundation`

Kairos — Adaptive and Generalizable Time Series Foundation Model with Mixture-of-Size Encoder.

## For Beginners

Kairos is a time series forecasting model that automatically
adjusts how it reads data based on complexity. In simple, steady periods it takes big
chunks at a time (like skimming a boring chapter), but in volatile periods it zooms in
and reads fine details (like carefully studying a plot twist). This adaptive approach
makes it both efficient and accurate across different types of data.

## How It Works

Kairos uses a Mixture-of-Size Encoder that adaptively tokenizes input at multiple
granularities based on local information density. A learned router decides which patch
size is optimal for each segment, making the model parameter-efficient and adaptive.

**Reference:** "Kairos: Towards Adaptive and Generalizable Time Series Foundation Models", 2025.
https://arxiv.org/abs/2509.25826

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Kairos(NeuralNetworkArchitecture<>,KairosOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a Kairos model in native mode. |
| `Kairos(NeuralNetworkArchitecture<>,String,KairosOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a Kairos model using a pretrained ONNX model. |

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

