---
title: "PatchTST<T>"
description: "PatchTST (Patch Time Series Transformer) neural network for long-term time series forecasting."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Forecasting.Transformers`

PatchTST (Patch Time Series Transformer) neural network for long-term time series forecasting.

## For Beginners

PatchTST treats a time series like a sentence: it breaks the
data into "patches" (chunks of consecutive values) just as a sentence is split into words.
A transformer then processes these patches to make predictions. This patching trick reduces
computational cost dramatically while capturing long-range patterns. Each variable in a
multivariate series is processed independently, which surprisingly improves accuracy.

## How It Works

PatchTST is a state-of-the-art transformer model for long-term time series forecasting.
It introduces patching (dividing time series into segments) and channel independence
to achieve efficient and accurate forecasting.

Reference: Nie et al., "A Time Series is Worth 64 Words: Long-term Forecasting
with Transformers", ICLR 2023. https://arxiv.org/abs/2211.14730

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PatchTST(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Boolean,Boolean,Double,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,PatchTSTOptions<>)` | Creates a PatchTST network using native library layers. |
| `PatchTST(NeuralNetworkArchitecture<>,String,Int32,Int32,Int32,Int32,Int32,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,PatchTSTOptions<>)` | Creates a PatchTST network using pretrained ONNX model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsChannelIndependent` |  |
| `PatchSize` |  |
| `Stride` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyInstanceNormalization(Tensor<>)` | Applies instance normalization (RevIN) during inference for distribution shift handling. |
| `ApplyRevIN(Tensor<>,Boolean)` | Applies Reversible Instance Normalization (RevIN) to handle distribution shift. |
| `AutoregressiveForecast(Tensor<>,Int32)` | Generates multi-step forecasts iteratively (autoregressive forecasting). |
| `CalculateInstanceMean(Tensor<>)` | Calculates the mean value for each feature across the time dimension. |
| `CalculateInstanceStd(Tensor<>,Tensor<>)` | Calculates the standard deviation for each feature across the time dimension. |
| `CalculateNumPatches` | Calculates the number of patches that will be created from the input sequence. |
| `ConcatenatePredictions(List<Tensor<>>,Int32)` | Concatenates multiple prediction tensors into a single long forecast. |
| `CreateNewInstance` | Creates a new instance of this model with the same configuration. |
| `CreatePatches(Tensor<>)` | Divides a time series sequence into overlapping patches. |
| `CreatePositionalEncoding(Int32,Int32)` | Creates sinusoidal positional encodings for the patches. |
| `DeserializeModelSpecificData(BinaryReader)` | Reads network-specific configuration data during deserialization. |
| `Dispose(Boolean)` | Releases resources used by the PatchTST model. |
| `Evaluate(Tensor<>,Tensor<>)` | Evaluates the model's forecasting performance on test data. |
| `ExtractChannel(Tensor<>,Int32,Int32)` | Extracts a single feature channel from the input tensor. |
| `ExtractLayerReferences` | Extracts references to specific layers from the layer collection for direct access. |
| `Forecast(Tensor<>,Double[])` | Generates forecasts with optional uncertainty quantification. |
| `ForecastNative(Tensor<>,Double[])` | Generates forecasts using native C# layers. |
| `ForecastOnnx(Tensor<>)` | Generates forecasts using a pretrained ONNX model. |
| `Forward(Tensor<>)` | Performs the forward pass through the PatchTST network. |
| `GetFinancialMetrics` | Gets financial-specific metrics from the model. |
| `GetModelMetadata` | Gets metadata about the model for serialization and inspection. |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the neural network layers for PatchTST. |
| `ProcessAllChannels(Tensor<>)` | Processes all feature channels together through the network (channel-dependent mode). |
| `ProcessChannelIndependent(Tensor<>)` | Processes each feature channel independently through the PatchTST network. |
| `ProcessSingleChannel(Tensor<>)` | Processes a single feature channel through the PatchTST layers. |
| `SerializeModelSpecificData(BinaryWriter)` | Writes network-specific configuration data during serialization. |
| `ShiftInputWithPredictions(Tensor<>,Tensor<>,Int32)` | Shifts the input window forward in time by incorporating new predictions. |
| `TrainCore(Tensor<>,Tensor<>,Tensor<>)` | Trains the model on a single batch of input-output pairs. |
| `UpdateParameters(Vector<>)` | Updates the model's parameters from a flat parameter vector. |
| `ValidateCustomLayers(List<ILayer<>>)` | Validates that custom layers meet PatchTST's architectural requirements. |
| `ValidateInputShape(Tensor<>)` | Validates the input tensor shape for PatchTST. |
| `ValidateParameters(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32)` | Executes ValidateParameters for the PatchTST. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_channelIndependent` | Whether to use channel-independent mode. |
| `_dropout` | Dropout rate. |
| `_encoderLayers` | Transformer encoder layers. |
| `_feedForwardDimension` | The feedforward dimension. |
| `_finalNorm` | Final layer normalization. |
| `_instanceMean` | Instance normalization mean (for RevIN). |
| `_instanceStd` | Instance normalization standard deviation (for RevIN). |
| `_modelDimension` | The model dimension. |
| `_numHeads` | The number of attention heads. |
| `_numLayers` | The number of transformer layers. |
| `_optimizer` | The optimizer for training. |
| `_outputProjection` | Output projection layer. |
| `_patchEmbedding` | Patch embedding layer. |
| `_patchSize` | The patch size. |
| `_positionalEncoding` | Positional encoding for patches. |
| `_stride` | The stride between patches. |
| `_useInstanceNormalization` | Whether to use instance normalization (RevIN). |

