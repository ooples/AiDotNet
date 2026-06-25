---
title: "ITransformer<T>"
description: "iTransformer (Inverted Transformer) neural network for time series forecasting."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Forecasting.Transformers`

iTransformer (Inverted Transformer) neural network for time series forecasting.

## For Beginners

Traditional transformers for time series treat each time step as a "word".
iTransformer flips this - it treats each variable (like price, volume) as a "word". This way,
the attention mechanism learns how different variables relate to each other, which is often
more useful for forecasting than just looking at temporal patterns.

## How It Works

iTransformer inverts the traditional transformer approach by treating each variable (channel)
as a token instead of each time step. This allows the model to learn cross-variable dependencies
more effectively through the attention mechanism.

**Reference:** Liu et al., "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting",
ICLR 2024. https://arxiv.org/abs/2310.06625

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ITransformer(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Boolean,Double,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,ITransformerOptions<>)` | Creates an iTransformer network using native library layers. |
| `ITransformer(NeuralNetworkArchitecture<>,String,Int32,Int32,Int32,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,ITransformerOptions<>)` | Creates an iTransformer network using a pretrained ONNX model. |

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
| `ApplyInstanceNormalization(Tensor<>)` | Applies instance normalization (RevIN) during inference for distribution shift handling. |
| `ApplyRevIN(Tensor<>,Boolean)` | Applies Reversible Instance Normalization (RevIN) to handle distribution shift. |
| `AutoregressiveForecast(Tensor<>,Int32)` | Generates multi-step forecasts iteratively (autoregressive forecasting). |
| `CalculateInstanceMean(Tensor<>)` | Calculates the mean value for each feature across the time dimension. |
| `CalculateInstanceStd(Tensor<>,Tensor<>)` | Calculates the standard deviation for each feature across the time dimension. |
| `ConcatenatePredictions(List<Tensor<>>,Int32)` | Concatenates multiple prediction tensors into a single long forecast. |
| `CreateNewInstance` | Creates a new instance of this model with the same configuration. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Reads network-specific configuration data during deserialization. |
| `Dispose(Boolean)` | Releases resources used by the iTransformer model. |
| `Evaluate(Tensor<>,Tensor<>)` | Evaluates the model's forecasting performance on test data. |
| `ExtractLayerReferences` | Extracts references to specific layers from the layer collection for direct access. |
| `Forecast(Tensor<>,Double[])` | Generates forecasts with optional uncertainty quantification. |
| `ForecastNative(Tensor<>,Double[])` | Generates forecasts using native C# layers. |
| `ForecastOnnx(Tensor<>)` | Generates forecasts using a pretrained ONNX model. |
| `Forward(Tensor<>)` | Performs the forward pass through the iTransformer network. |
| `ForwardNativeForTraining(Tensor<>)` | Training-mode forward: calls `Tensor{` directly so attention dropout and the inverted-variable projection stay in training mode under the gradient tape. |
| `GetFinancialMetrics` | Gets financial-specific metrics from the model. |
| `GetModelMetadata` | Gets metadata about the model for serialization and inspection. |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the neural network layers for iTransformer. |
| `InvertInput(Tensor<>)` | Inverts the input tensor by transposing time and feature dimensions. |
| `PredictCore(Tensor<>)` | Makes a prediction using the input tensor. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Writes network-specific configuration data during serialization. |
| `ShiftInputWithPredictions(Tensor<>,Tensor<>,Int32)` | Shifts the input window forward in time by incorporating new predictions. |
| `Train(Tensor<>,Tensor<>)` | Trains the model on a single batch of input-output pairs. |
| `UninvertOutput(Tensor<>)` | Un-inverts the output tensor by transposing back to standard layout. |
| `UpdateParameters(Vector<>)` | Updates the model's parameters from a flat parameter vector. |
| `ValidateCustomLayers(List<ILayer<>>)` | Validates that custom layers meet iTransformer's architectural requirements. |
| `ValidateParameters(Int32,Int32,Int32,Int32,Int32,Int32)` | Validates constructor parameters. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_dropout` | Dropout rate. |
| `_encoderLayers` | Transformer encoder layers for cross-variable attention. |
| `_feedForwardDimension` | The feedforward dimension. |
| `_finalNorm` | Final layer normalization. |
| `_instanceMean` | Instance normalization mean (for RevIN). |
| `_instanceStd` | Instance normalization standard deviation (for RevIN). |
| `_lossFunction` | The loss function for training. |
| `_modelDimension` | The model dimension. |
| `_numFeatures` | The number of input features. |
| `_numHeads` | The number of attention heads. |
| `_numLayers` | The number of transformer layers. |
| `_optimizer` | The optimizer for training. |
| `_outputProjection` | Output projection layer. |
| `_predictionHorizon` | The prediction horizon. |
| `_sequenceLength` | The input sequence length. |
| `_useInstanceNormalization` | Whether to use instance normalization (RevIN). |
| `_useNativeMode` | Indicates whether this network uses native layers (true) or ONNX model (false). |
| `_variateEmbedding` | Variate embedding layer that embeds each variable's time series. |

