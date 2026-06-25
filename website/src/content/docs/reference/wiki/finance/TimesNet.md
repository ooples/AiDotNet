---
title: "TimesNet<T>"
description: "TimesNet (Temporal 2D-Variation Modeling) neural network for time series analysis."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Forecasting.Transformers`

TimesNet (Temporal 2D-Variation Modeling) neural network for time series analysis.

## For Beginners

TimesNet thinks about time series data like a calendar:

- Instead of just seeing days in a row (1D), it arranges them into weeks and months (2D)
- This makes it easy to see patterns like "every Monday sales drop" or "end of month peaks"
- It automatically discovers what time scales matter most (daily, weekly, quarterly, etc.)

Key innovations:

- **Period Discovery:** Uses FFT to automatically find dominant periods
- **2D Transformation:** Reshapes 1D time series into 2D based on periods
- **Inception Module:** Multi-scale convolutions capture patterns at different granularities

## How It Works

TimesNet transforms 1D time series into 2D tensors based on automatically discovered
periods, then applies 2D convolutions to capture both intra-period and inter-period
variations. This approach is particularly effective for time series with multiple
periodic patterns.

**Reference:** Wu et al., "TimesNet: Temporal 2D-Variation Modeling for General
Time Series Analysis", ICLR 2023. https://arxiv.org/abs/2210.02186

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TimesNet(NeuralNetworkArchitecture<>,String,TimesNetOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a TimesNet network using pretrained ONNX model. |
| `TimesNet(NeuralNetworkArchitecture<>,TimesNetOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a TimesNet network in native mode for training from scratch. |

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
| `AdjustToPredictionHorizon(Tensor<>)` | Trims the sequence dimension to `_predictionHorizon`, taking the LAST `pred_len` timesteps per Wu et al. |
| `ApplyInstanceNormalization(Tensor<>)` |  |
| `ApplyRevIN(Tensor<>,Boolean)` | Applies RevIN normalization/denormalization. |
| `AutoregressiveForecast(Tensor<>,Int32)` |  |
| `ConcatenatePredictions(List<Tensor<>>,Int32)` | Concatenates multiple prediction tensors. |
| `CreateNewInstance` | Creates a new instance of this model with the same configuration. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Reads TimesNet-specific configuration during deserialization. |
| `Dispose(Boolean)` | Disposes resources used by the TimesNet model. |
| `Evaluate(Tensor<>,Tensor<>)` |  |
| `ExtractLayerReferences` | Extracts references to specific layers from the layer collection. |
| `Forecast(Tensor<>,Double[])` |  |
| `ForecastNative(Tensor<>)` | Performs native mode forecasting. |
| `ForecastOnnx(Tensor<>)` | Performs ONNX mode forecasting. |
| `Forward(Tensor<>)` | Performs the forward pass through the TimesNet network. |
| `ForwardNativeForTraining(Tensor<>)` | Training-mode forward: calls `Tensor{` directly so `_dropoutLayers` between the Inception blocks fire during backprop. |
| `GetFinancialMetrics` |  |
| `GetModelMetadata` |  |
| `GetNamedLayerActivations(Tensor<>)` | Captures named activations along TimesNet's actual forward path. |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the neural network layers for TimesNet. |
| `PredictCore(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` | Writes TimesNet-specific configuration during serialization. |
| `SetTrainingMode(Boolean)` | Propagates training mode to every layer. |
| `ShiftInputWithPredictions(Tensor<>,Tensor<>,Int32)` | Shifts input and appends predictions for autoregressive forecasting. |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |
| `ValidateCustomLayers(List<ILayer<>>)` | Validates that custom layers meet TimesNet's architectural requirements. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_convKernelSize` | Convolution kernel size. |
| `_convLayers` | Convolutional layers in TimesBlocks. |
| `_dropout` | Dropout rate for regularization. |
| `_dropoutLayers` | Dropout layers. |
| `_embeddingLayer` | Embedding layer for projecting input features. |
| `_feedForwardDimension` | The feedforward network dimension. |
| `_ffnLayers` | Feedforward layers in TimesBlocks. |
| `_finalNorm` | Final layer normalization. |
| `_instanceMean` | Instance normalization mean (for RevIN). |
| `_instanceStd` | Instance normalization standard deviation (for RevIN). |
| `_lossFunction` | The loss function for training. |
| `_modelDimension` | The model dimension (embedding size). |
| `_normLayers` | Layer normalization layers. |
| `_numFeatures` | The number of input features. |
| `_numLayers` | Number of TimesBlock layers. |
| `_optimizer` | The optimizer for training. |
| `_outputProjection` | Output projection layer. |
| `_predictionHorizon` | The prediction horizon. |
| `_sequenceLength` | The input sequence length. |
| `_topK` | Number of dominant periods to discover. |
| `_useInstanceNormalization` | Whether to use instance normalization (RevIN). |
| `_useNativeMode` | Indicates whether this network uses native layers (true) or ONNX model (false). |

