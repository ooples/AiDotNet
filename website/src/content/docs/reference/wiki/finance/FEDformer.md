---
title: "FEDformer<T>"
description: "FEDformer (Frequency Enhanced Decomposed Transformer) for long-term time series forecasting."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Forecasting.Transformers`

FEDformer (Frequency Enhanced Decomposed Transformer) for long-term time series forecasting.

## For Beginners

FEDformer is like analyzing music by frequencies instead of individual
notes. Standard transformers look at every time step (expensive), but FEDformer converts to
frequency domain where patterns are simpler and operations are faster.

Key innovations:

- **Frequency Attention:** Uses Fourier/Wavelet transforms for O(N) complexity
- **Decomposition:** Separates trend (overall direction) from seasonal (repeating patterns)
- **Random Mode Selection:** Keeps only important frequencies for efficiency

## How It Works

FEDformer achieves linear complexity O(N) by performing attention in the frequency domain
using Fourier or Wavelet transforms. It also decomposes time series into trend and seasonal
components for better interpretability and forecasting accuracy.

**Reference:** Zhou et al., "FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting",
ICML 2022. https://arxiv.org/abs/2201.12740

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FEDformer(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Boolean,Double,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,FEDformerOptions<>)` | Creates a FEDformer network using native library layers. |
| `FEDformer(NeuralNetworkArchitecture<>,String,Int32,Int32,Int32,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,FEDformerOptions<>)` | Creates a FEDformer network using a pretrained ONNX model. |

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
| `ApplyInstanceNormalization(Tensor<>)` |  |
| `ApplyRevIN(Tensor<>,Boolean)` | Applies RevIN (Reversible Instance Normalization). |
| `AutoregressiveForecast(Tensor<>,Int32)` |  |
| `CalculateInstanceMean(Tensor<>)` | Calculates the mean for each feature. |
| `CalculateInstanceStd(Tensor<>,Tensor<>)` | Calculates the standard deviation for each feature. |
| `ConcatenatePredictions(List<Tensor<>>,Int32)` | Concatenates multiple predictions. |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Dispose(Boolean)` |  |
| `Evaluate(Tensor<>,Tensor<>)` |  |
| `ExtractLayerReferences` | Extracts references to specific layers for direct access. |
| `Forecast(Tensor<>,Double[])` |  |
| `ForecastNative(Tensor<>,Double[])` | Forecasts using native layers. |
| `ForecastOnnx(Tensor<>)` | Forecasts using ONNX model. |
| `Forward(Tensor<>)` | Performs the forward pass through the FEDformer network. |
| `ForwardNativeForTraining(Tensor<>)` | Training-mode forward: calls `Tensor{` directly so the frequency-enhanced attention runs under its training-mode behavior (e.g. |
| `GetFinancialMetrics` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the neural network layers for FEDformer. |
| `PredictCore(Tensor<>)` | Makes a prediction using the input tensor. |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `ShiftInputWithPredictions(Tensor<>,Tensor<>,Int32)` | Shifts input with predictions for autoregressive forecasting. |
| `Train(Tensor<>,Tensor<>)` | Trains the model on a single batch. |
| `UpdateParameters(Vector<>)` |  |
| `ValidateCustomLayers(List<ILayer<>>)` | Validates that custom layers meet FEDformer's requirements. |
| `ValidateParameters(Int32,Int32,Int32,Int32,Int32,Int32,Int32)` | Validates constructor parameters. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_decoderLayers` | Decoder layers for generating predictions. |
| `_dropout` | Dropout rate. |
| `_encoderLayers` | Encoder layers for frequency-enhanced attention. |
| `_feedForwardDimension` | The feedforward dimension. |
| `_finalNorm` | Final layer normalization. |
| `_inputEmbedding` | Input embedding layer. |
| `_instanceMean` | Instance normalization mean (for RevIN). |
| `_instanceStd` | Instance normalization standard deviation (for RevIN). |
| `_lossFunction` | The loss function for training. |
| `_modelDimension` | The model dimension. |
| `_movingAverageKernel` | Moving average kernel size for decomposition. |
| `_numDecoderLayers` | The number of decoder layers. |
| `_numEncoderLayers` | The number of encoder layers. |
| `_numFeatures` | The number of input features. |
| `_numHeads` | The number of attention heads. |
| `_numModes` | Number of frequency modes to use in attention. |
| `_optimizer` | The optimizer for training. |
| `_outputProjection` | Output projection layer. |
| `_predictionHorizon` | The prediction horizon. |
| `_sequenceLength` | The input sequence length. |
| `_useInstanceNormalization` | Whether to use instance normalization (RevIN). |
| `_useNativeMode` | Indicates whether this network uses native layers (true) or ONNX model (false). |

