---
title: "ETSformer<T>"
description: "ETSformer: Exponential Smoothing Transformer for time series forecasting."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Forecasting.Transformers`

ETSformer: Exponential Smoothing Transformer for time series forecasting.

## For Beginners

ETSformer is like having a smart forecaster that can:

1. **See the Level**: The overall value of the time series (like average price level)
2. **See the Trend**: Whether values are going up or down over time
3. **See Seasonality**: Repeating patterns (like daily or weekly cycles)

Unlike black-box models, ETSformer lets you inspect each component to understand
WHY it makes certain predictions. This is valuable in finance where explainability matters.

## How It Works

ETSformer combines classical exponential smoothing (ETS) methods with transformer
attention mechanisms to create an interpretable time series forecasting model.
It explicitly models level, trend, and seasonal components.

**Key Innovation:** The exponential smoothing attention mechanism applies learnable
decay factors, giving more weight to recent observations while still considering
historical patterns through the transformer architecture.

**Reference:** Woo et al., "ETSformer: Exponential Smoothing Transformers for
Time-series Forecasting", 2022. https://arxiv.org/abs/2202.01381

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ETSformer(NeuralNetworkArchitecture<>,ETSformerOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates an ETSformer model in native C# mode for training and inference. |
| `ETSformer(NeuralNetworkArchitecture<>,String,ETSformerOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates an ETSformer model in ONNX inference mode using a pre-trained model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsChannelIndependent` |  |
| `NumFeatures` |  |
| `PatchSize` |  |
| `PredictionHorizon` |  |
| `SequenceLength` |  |
| `Stride` |  |
| `SupportsTraining` | Gets whether this model supports training (native mode only). |
| `UseNativeMode` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyInstanceNormalization(Tensor<>)` |  |
| `AutoregressiveForecast(Tensor<>,Int32)` |  |
| `ConcatenatePredictions(List<Tensor<>>,Int32)` | Concatenates prediction tensors and trims to requested steps. |
| `ConvertFromOnnxTensor(Tensor<Single>)` | Converts an ONNX tensor back to our tensor type. |
| `ConvertToFloatArray(Tensor<>)` | Converts our tensor to a float array for ONNX. |
| `CreateNewInstance` | Creates a new instance of this model with the same configuration. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes model-specific data when loading. |
| `Dispose(Boolean)` | Disposes of managed resources. |
| `Evaluate(Tensor<>,Tensor<>)` |  |
| `ExtractLayerReferences` | Extracts references to specific layers for direct access during forward/backward passes. |
| `Forecast(Tensor<>,Double[])` |  |
| `ForecastNative(Tensor<>)` | Runs inference using the native C# implementation. |
| `ForecastOnnx(Tensor<>)` | Runs inference using the ONNX Runtime. |
| `Forward(Tensor<>)` | Performs the forward pass through all layers. |
| `GetFinancialMetrics` |  |
| `GetModelMetadata` | Gets metadata describing the model. |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the neural network layers for the ETSformer architecture. |
| `PredictCore(Tensor<>)` | Makes predictions for the given input tensor. |
| `ReshapeOutput(Tensor<>)` | Reshapes the output to the expected dimensions. |
| `ReverseInstanceNormalization(Tensor<>)` | Reverses instance normalization on the output. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes model-specific data for saving. |
| `ShiftInputWithPredictions(Tensor<>,Tensor<>,Int32)` | Shifts input tensor and appends predictions for autoregressive forecasting. |
| `Train(Tensor<>,Tensor<>)` | Trains the model on a batch of data. |
| `UpdateParameters(Vector<>)` | Updates the model parameters using the provided gradient vector. |
| `ValidateCustomLayers(List<ILayer<>>)` | Validates that custom layers meet ETSformer requirements. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_decoderAttentionLayers` | Reference to decoder attention layers. |
| `_dropout` | The dropout rate. |
| `_encoderAttentionLayers` | Reference to encoder attention layers. |
| `_inputEmbedding` | Reference to the input embedding layer. |
| `_instanceMean` | Instance normalization mean (for RevIN). |
| `_instanceStd` | Instance normalization standard deviation (for RevIN). |
| `_lossFunction` | The loss function used for training. |
| `_modelDimension` | The model dimension (embedding size). |
| `_numDecoderLayers` | The number of decoder layers. |
| `_numEncoderLayers` | The number of encoder layers. |
| `_numFeatures` | The number of input features. |
| `_numHeads` | The number of attention heads. |
| `_optimizer` | The optimizer used for training (gradient-based parameter updates). |
| `_outputProjection` | Reference to the output projection layer. |
| `_predictionHorizon` | The prediction horizon. |
| `_sequenceLength` | The input sequence length. |
| `_topK` | Top-K frequencies for seasonal decomposition. |
| `_useInstanceNormalization` | Whether to use instance normalization (RevIN). |
| `_useNativeMode` | Indicates whether the model operates in native C# mode (true) or ONNX inference mode (false). |

