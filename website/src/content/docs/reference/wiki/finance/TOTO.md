---
title: "TOTO<T>"
description: "TOTO — Datadog's Time Series Foundation Model for Observability."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Forecasting.Foundation`

TOTO — Datadog's Time Series Foundation Model for Observability.

## For Beginners

TOTO is a specialized model from Datadog, trained on 1 trillion
data points from real-world server monitoring. It excels at predicting system metrics
like CPU usage, memory consumption, and request latency. If you monitor infrastructure
and need to forecast capacity or detect anomalies before they cause outages, TOTO is
purpose-built for that domain.

## How It Works

TOTO is Datadog's domain-specific time series foundation model optimized for IT operations,
infrastructure monitoring, and observability. Pre-trained on 1 trillion data points from
the Datadog observability platform, it excels at SRE metrics and anomaly detection.

**Reference:** Datadog, "Introducing Toto: A state-of-the-art time series foundation model", 2025.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TOTO(NeuralNetworkArchitecture<>,String,TOTOOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a TOTO model using a pretrained ONNX model. |
| `TOTO(NeuralNetworkArchitecture<>,TOTOOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a TOTO model in native mode for training or fine-tuning. |

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
| `GetFinancialMetrics` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `PredictCore(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

