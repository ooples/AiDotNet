---
title: "RWKVForecaster<T>"
description: "RWKV (Receptance Weighted Key Value) implementation for time series forecasting."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Forecasting.StateSpace`

RWKV (Receptance Weighted Key Value) implementation for time series forecasting.

## For Beginners

RWKV is a unique architecture for sequence modeling:

**The Key Innovation:**
RWKV can be computed two ways:

1. As a parallel operation during training (like a Transformer) — fast on GPUs
2. As a recurrence during inference (like an RNN) — constant memory per token

**Architecture Components:**

- Time mixing: WKV (Weighted Key Value) attention with learned exponential decay
- Channel mixing: Feed-forward network with gating mechanism
- Token shift: Efficiently mixes current and previous token information

**For Time Series:**

- Linear complexity enables processing very long historical windows
- Constant memory during autoregressive forecasting
- Multi-head structure captures diverse temporal patterns

## How It Works

RWKV combines the efficient parallelizable training of Transformers with the efficient
inference of RNNs, achieving linear complexity for both training and inference.
This forecasting variant uses stacked RWKV layers for time series prediction.

**Reference:** Peng et al., "RWKV: Reinventing RNNs for the Transformer Era", 2023.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RWKVForecaster(NeuralNetworkArchitecture<>,RWKVForecastingOptions<>,Int32,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new instance in native mode for training. |
| `RWKVForecaster(NeuralNetworkArchitecture<>,String,RWKVForecastingOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new instance using an ONNX pretrained model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsChannelIndependent` |  |
| `NumFeatures` |  |
| `NumHeads` | Gets the number of RWKV heads. |
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
| `ConcatenatePredictions(List<Tensor<>>,Int32)` |  |
| `CreateNewInstance` |  |
| `DenormalizeForecast(Tensor<>)` | RevIN reverse step (Kim et al. |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Dispose(Boolean)` |  |
| `Evaluate(Tensor<>,Tensor<>)` |  |
| `Forecast(Tensor<>,Double[])` |  |
| `ForecastOnnx(Tensor<>)` |  |
| `ForwardNativeForTraining(Tensor<>)` | Training-mode forward. |
| `GetFinancialMetrics` |  |
| `GetModelMetadata` |  |
| `GetNamedLayerActivations(Tensor<>)` | Captures the per-layer activations along the model's real forward path. |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `PredictCore(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `ShiftInputWithPredictions(Tensor<>,Tensor<>,Int32)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |
| `ValidateCustomLayers(List<ILayer<>>)` |  |

