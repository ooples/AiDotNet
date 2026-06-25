---
title: "Autoformer<T>"
description: "Autoformer (Decomposition Transformers with Auto-Correlation) neural network for time series forecasting."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Forecasting.Transformers`

Autoformer (Decomposition Transformers with Auto-Correlation) neural network for time series forecasting.

## For Beginners

Autoformer is like a music analyst who finds repeating patterns
by checking if the melody matches itself at different time delays. Instead of comparing
every note to every other note (O(L²)), it uses FFT to find these patterns in O(L log L) time.

Key innovations:

- **Auto-Correlation:** Finds periodic patterns using FFT instead of attention
- **Series Decomposition:** Separates trend from seasonal patterns at each layer
- **Progressive Decomposition:** Decomposition happens multiple times for accuracy

## How It Works

Autoformer replaces the traditional attention mechanism with auto-correlation, which finds
period-based dependencies efficiently using FFT. It also progressively decomposes the series
into trend and seasonal components at each layer.

**Reference:** Wu et al., "Autoformer: Decomposition Transformers with Auto-Correlation",
NeurIPS 2021. https://arxiv.org/abs/2106.13008

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Autoformer(NeuralNetworkArchitecture<>,AutoformerOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates an Autoformer network in native mode for training from scratch. |
| `Autoformer(NeuralNetworkArchitecture<>,String,AutoformerOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates an Autoformer network using a pretrained ONNX model. |

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
| `AddTensors(Tensor<>,Tensor<>)` | Adds two tensors element-wise. |
| `AdjustToPredictionHorizon(Tensor<>)` | Adjusts output to prediction horizon. |
| `ApplyInstanceNormalization(Tensor<>)` |  |
| `ApplyRevIN(Tensor<>,Boolean)` | Applies RevIN normalization/denormalization. |
| `AutoregressiveForecast(Tensor<>,Int32)` |  |
| `ConcatenatePredictions(List<Tensor<>>,Int32)` | Concatenates prediction tensors. |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Dispose(Boolean)` | Releases resources. |
| `Evaluate(Tensor<>,Tensor<>)` |  |
| `ExtractLayerReferences` | Extracts references to specific layers. |
| `Forecast(Tensor<>,Double[])` |  |
| `ForecastNative(Tensor<>,Double[])` | Performs native mode forecasting. |
| `ForecastOnnx(Tensor<>)` | Performs ONNX mode forecasting. |
| `Forward(Tensor<>)` | Performs the forward pass through all layers. |
| `ForwardNativeForTraining(Tensor<>)` | Training-mode forward: calls `Tensor{` so attention dropout and series-decomp noise (both gated on training mode) stay active under the gradient tape. |
| `GetFinancialMetrics` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeDecomposition(Tensor<>)` | Initializes decomposition components. |
| `InitializeLayers` | Initializes the layers for native mode operation. |
| `MovingAverage(Tensor<>,Int32)` | Computes moving average. |
| `PredictCore(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `SeriesDecomposition(Tensor<>)` | Performs series decomposition using moving average. |
| `ShiftInputWithPredictions(Tensor<>,Tensor<>,Int32)` | Shifts input and appends predictions. |
| `SubtractTensors(Tensor<>,Tensor<>)` | Subtracts tensor b from tensor a. |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |
| `ValidateCustomLayers(List<ILayer<>>)` | Validates custom layers provided by the user. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_decoderLayers` | Decoder layers for generating predictions. |
| `_encoderLayers` | Encoder layers with auto-correlation. |
| `_finalNorm` | Final layer normalization. |
| `_inputEmbedding` | Input embedding layer that projects features to model dimension. |
| `_instanceMean` | Instance normalization mean (for RevIN). |
| `_instanceStd` | Instance normalization standard deviation (for RevIN). |
| `_lossFunction` | The loss function for computing prediction errors. |
| `_optimizer` | The optimizer for training the model. |
| `_outputProjection` | Output projection layer. |
| `_seasonalComponent` | Seasonal component from decomposition. |
| `_trendComponent` | Trend component from decomposition. |
| `_useNativeMode` | Indicates whether this network uses native layers (true) or ONNX model (false). |

