---
title: "TFC<T>"
description: "TF-C — Time-Frequency Consistency for Self-Supervised Time Series."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Forecasting.Foundation`

TF-C — Time-Frequency Consistency for Self-Supervised Time Series.

## For Beginners

TF-C learns to understand time series by looking at the same
data in two ways: as a sequence of values over time, and as a set of frequencies (like
breaking a musical chord into individual notes). By training the model to agree on what
it sees from both perspectives, it learns robust patterns that work well for downstream
tasks like forecasting and classification.

## How It Works

TF-C learns time series representations by enforcing consistency between time-domain
and frequency-domain representations via contrastive learning, capturing both
temporal and spectral patterns. It uses dual CNN encoders with a shared projection head.

**Reference:** Zhang et al., "Self-Supervised Contrastive Pre-Training For Time Series via Time-Frequency Consistency", NeurIPS 2022.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TFC(NeuralNetworkArchitecture<>,String,TFCOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a TF-C model using a pretrained ONNX model. |
| `TFC(NeuralNetworkArchitecture<>,TFCOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a TF-C model in native mode for training or fine-tuning. |

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
| `ComputeContrastiveLoss(Tensor<>)` | Contrastive loss between time and frequency encoder outputs. |
| `ComputeContrastiveLossTape(Tensor<>)` | Tape-aware version of `Tensor{` that returns a `Tensor` (not a `T` scalar) so the gradient tape records every op between the encoder outputs and the final loss. |
| `ComputeFrequencyRepresentation(Tensor<>)` | Computes the DFT magnitude spectrum of the input time series. |
| `CreateNewInstance` |  |
| `DenormalizeForecast(Tensor<>)` | RevIN reverse step (Kim et al. |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Evaluate(Tensor<>,Tensor<>)` |  |
| `Forecast(Tensor<>,Double[])` |  |
| `ForwardNativeForTraining(Tensor<>)` | Training-mode forward. |
| `GetFinancialMetrics` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `PredictCore(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

