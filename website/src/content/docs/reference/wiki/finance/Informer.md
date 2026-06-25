---
title: "Informer<T>"
description: "Informer (Efficient Transformer for Long Sequence Forecasting) neural network."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Forecasting.Transformers`

Informer (Efficient Transformer for Long Sequence Forecasting) neural network.

## For Beginners

In regular transformers, every position looks at every other position
(O(n²) complexity), which is slow for long sequences. Informer is clever - it figures out
which positions are most "active" (have high variance in their attention scores) and only
computes attention for those.

Key innovations:

- **ProbSparse Attention:** Only compute attention for top-k important queries
- **Self-attention Distilling:** Progressively reduce sequence length with max-pooling
- **Generative Decoder:** Predict all future values at once, not step-by-step

## How It Works

Informer was the first transformer specifically designed for long sequence time series
forecasting. It achieved AAAI 2021 Best Paper for its innovations in efficient attention.

**Reference:** Zhou et al., "Informer: Beyond Efficient Transformer for Long Sequence
Time-Series Forecasting", AAAI 2021 (Best Paper). https://arxiv.org/abs/2012.07436

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Informer` | Creates an Informer model with default configuration for native training. |
| `Informer(NeuralNetworkArchitecture<>,InformerOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates an Informer network in native mode for training from scratch. |
| `Informer(NeuralNetworkArchitecture<>,String,InformerOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates an Informer network using a pretrained ONNX model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsChannelIndependent` |  |
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
| `ApplyDistilling(Tensor<>)` | Applies self-attention distilling. |
| `ApplyInstanceNormalization(Tensor<>)` |  |
| `ApplyRevIN(Tensor<>,Boolean)` | Applies RevIN normalization/denormalization. |
| `AutoregressiveForecast(Tensor<>,Int32)` |  |
| `ConcatenatePredictions(List<Tensor<>>,Int32)` | Concatenates prediction tensors. |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Dispose(Boolean)` | Releases resources. |
| `Evaluate(Tensor<>,Tensor<>)` |  |
| `ExtractLayerReferences` | Extracts references to specific layers. |
| `ExtractPrediction(Tensor<>)` | Extracts prediction portion from decoder output. |
| `Forecast(Tensor<>,Double[])` |  |
| `ForecastNative(Tensor<>,Double[])` | Performs native mode forecasting. |
| `ForecastOnnx(Tensor<>)` | Performs ONNX mode forecasting. |
| `Forward(Tensor<>)` | Performs the forward pass. |
| `GetFinancialMetrics` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the layers for native mode operation. |
| `PredictCore(Tensor<>)` |  |
| `PrepareDecoderInput(Tensor<>,Tensor<>)` | Prepares decoder input. |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `ShiftInputWithPredictions(Tensor<>,Tensor<>,Int32)` | Shifts input and appends predictions. |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |
| `ValidateCustomLayers(List<ILayer<>>)` | Validates custom layers. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_decoderLayers` | Decoder layers. |
| `_encoderLayers` | Encoder layers with ProbSparse attention. |
| `_finalNorm` | Final layer normalization. |
| `_inputEmbedding` | Input embedding layer. |
| `_instanceMean` | Instance normalization mean (for RevIN). |
| `_instanceStd` | Instance normalization standard deviation (for RevIN). |
| `_lossFunction` | The loss function for computing prediction errors. |
| `_optimizer` | The optimizer for training the model. |
| `_outputProjection` | Output projection layer. |
| `_useNativeMode` | Indicates whether this network uses native layers (true) or ONNX model (false). |

