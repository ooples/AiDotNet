---
title: "TimeMoE<T>"
description: "Time-MoE — Billion-Scale Time Series Foundation Models with Mixture of Experts."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Forecasting.Foundation`

Time-MoE — Billion-Scale Time Series Foundation Models with Mixture of Experts.

## For Beginners

Time-MoE is the first time series model to reach billions of
parameters by using a Mixture of Experts approach. Instead of one giant network processing
every input, it has many specialized "expert" sub-networks and a router that picks the best
2-3 experts for each data point. This means only a fraction of the parameters are active
at any time, making it efficient despite its massive total size.

## How It Works

Time-MoE is the first billion-scale time series foundation model, using sparse Mixture
of Experts for efficient scaling up to 2.4B total parameters (~300M active per token).
It uses a decoder-only transformer where each feed-forward layer is replaced by an
MoE layer with a learned router.

**Reference:** Shi et al., "Time-MoE: Billion-Scale Time Series Foundation Models
with Mixture of Experts", ICLR 2025. https://openreview.net/forum?id=e1wDDFmlVu

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TimeMoE(NeuralNetworkArchitecture<>,String,TimeMoEOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a Time-MoE model using a pretrained ONNX model. |
| `TimeMoE(NeuralNetworkArchitecture<>,TimeMoEOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a Time-MoE model in native mode. |

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

