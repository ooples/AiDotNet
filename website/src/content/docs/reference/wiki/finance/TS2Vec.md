---
title: "TS2Vec<T>"
description: "TS2Vec — Contrastive Learning of Universal Time Series Representations."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Forecasting.Foundation`

TS2Vec — Contrastive Learning of Universal Time Series Representations.

## For Beginners

TS2Vec creates a universal "fingerprint" for time series data
at any time scale. It works by showing the model two different views of the same data
(like seeing a city from two angles) and training it to recognize they represent the same
thing. The resulting representations can be used for forecasting, classification, or
anomaly detection without task-specific retraining.

## How It Works

TS2Vec learns universal time series representations via hierarchical contrastive learning
across augmented context views, producing contextual representations at arbitrary granularities.
It uses a dilated CNN encoder with temporal and instance contrastive losses.

**Reference:** Yue et al., "TS2Vec: Towards Universal Representation of Time Series", AAAI 2022.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TS2Vec(NeuralNetworkArchitecture<>,String,TS2VecOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a TS2Vec model using a pretrained ONNX model. |
| `TS2Vec(NeuralNetworkArchitecture<>,TS2VecOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a TS2Vec model in native mode for training or fine-tuning. |

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
| `DenormalizeForecast(Tensor<>)` | RevIN reverse step: restores each instance's mean/std to the forecast so it is expressed on the input's original scale (Kim et al. |
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

