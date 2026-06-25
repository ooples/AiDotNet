---
title: "Crossformer<T>"
description: "Crossformer (Cross-Dimension Transformer) neural network for multivariate time series forecasting."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Forecasting.Transformers`

Crossformer (Cross-Dimension Transformer) neural network for multivariate time series forecasting.

## For Beginners

Crossformer is designed for data where:

- Multiple variables influence each other (like stocks in the same sector)
- Both time patterns AND variable relationships matter
- You need to capture how one variable's past affects another's future

Key innovations:

- **Two-Stage Attention (TSA):** Alternates between cross-time and cross-dimension attention
- **Dimension-Segment Embedding:** Embeds each variable-segment pair
- **Hierarchical Structure:** Processes at multiple time scales

## How It Works

Crossformer uses a two-stage attention mechanism that captures both temporal dependencies
and cross-variable relationships simultaneously. It's particularly effective for multivariate
time series where different variables interact in complex ways.

**Reference:** Zhang et al., "Crossformer: Transformer Utilizing Cross-Dimension
Dependency for Multivariate Time Series Forecasting", ICLR 2023.
https://openreview.net/forum?id=vSVLM2j9eie

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Crossformer(NeuralNetworkArchitecture<>,CrossformerOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a Crossformer network in native mode for training from scratch. |
| `Crossformer(NeuralNetworkArchitecture<>,String,CrossformerOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a Crossformer network using pretrained ONNX model. |

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
| `AddResidualConnection(Tensor<>,Tensor<>)` | Adds a residual connection between input and processed output. |
| `AdjustToPredictionHorizon(Tensor<>)` | Adjusts output to match prediction horizon. |
| `ApplyInstanceNormalization(Tensor<>)` |  |
| `ApplyRevIN(Tensor<>,Boolean)` | Applies RevIN normalization/denormalization. |
| `AutoregressiveForecast(Tensor<>,Int32)` |  |
| `ConcatenatePredictions(List<Tensor<>>,Int32)` | Concatenates multiple prediction tensors. |
| `CreateNewInstance` | Creates a new instance of this model with the same configuration. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Reads Crossformer-specific configuration during deserialization. |
| `Dispose(Boolean)` | Disposes resources used by the Crossformer model. |
| `Evaluate(Tensor<>,Tensor<>)` |  |
| `ExtractLayerReferences` | Extracts references to specific layers from the layer collection. |
| `Forecast(Tensor<>,Double[])` |  |
| `ForecastNative(Tensor<>)` | Performs native mode forecasting. |
| `ForecastOnnx(Tensor<>)` | Performs ONNX mode forecasting. |
| `Forward(Tensor<>)` | Performs the forward pass through the Crossformer network. |
| `GetFinancialMetrics` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the neural network layers for Crossformer. |
| `PredictCore(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` | Writes Crossformer-specific configuration during serialization. |
| `ShiftInputWithPredictions(Tensor<>,Tensor<>,Int32)` | Shifts input and appends predictions for autoregressive forecasting. |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |
| `ValidateCustomLayers(List<ILayer<>>)` | Validates that custom layers meet Crossformer's architectural requirements. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_crossDimAttentionLayers` | Cross-Dimension attention layers. |
| `_crossTimeAttentionLayers` | Cross-Time attention layers. |
| `_dropout` | Dropout rate for regularization. |
| `_dropoutLayers` | Dropout layers between attention stages. |
| `_dseLayer` | Dimension-Segment Embedding layer. |
| `_finalNorm` | Final layer normalization. |
| `_instanceMean` | Instance normalization mean (for RevIN). |
| `_instanceStd` | Instance normalization standard deviation (for RevIN). |
| `_lossFunction` | The loss function for training. |
| `_modelDimension` | The model dimension (embedding size). |
| `_numFeatures` | The number of input features. |
| `_numHeads` | Number of attention heads. |
| `_numLayers` | Number of transformer layers. |
| `_optimizer` | The optimizer for training. |
| `_outputProjection` | Output projection layer. |
| `_predictionHorizon` | The prediction horizon. |
| `_segmentLength` | The segment length for cross-time attention. |
| `_sequenceLength` | The input sequence length. |
| `_useInstanceNormalization` | Whether to use instance normalization (RevIN). |
| `_useNativeMode` | Indicates whether this network uses native layers (true) or ONNX model (false). |

