---
title: "NonStationaryTransformer<T>"
description: "Non-stationary Transformer neural network for time series forecasting with changing statistics."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Forecasting.Transformers`

Non-stationary Transformer neural network for time series forecasting with changing statistics.

## For Beginners

Real-world time series data often has changing statistical properties:

- Stock prices trend up or down over time
- Sales data has seasonal patterns that change intensity
- Energy consumption varies with economic conditions

Traditional approaches normalize this data (make it "stationary"), but this can lose important
information. Non-stationary Transformer:

1. **Series Stationarization:** Normalizes data for better attention computation
2. **De-stationary Attention:** Learns to restore original statistical properties

This gives you the best of both worlds - good pattern recognition AND preserved data characteristics.

## How It Works

Non-stationary Transformer addresses the over-stationarization problem in time series forecasting
by proposing Series Stationarization and De-stationary Attention mechanisms that preserve
original data characteristics while benefiting from attention's pattern recognition.

**Reference:** Liu et al., "Non-stationary Transformers: Exploring the Stationarity
in Time Series Forecasting", NeurIPS 2022. https://arxiv.org/abs/2205.14415

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NonStationaryTransformer(NeuralNetworkArchitecture<>,NonStationaryTransformerOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a new Non-stationary Transformer instance for native training mode. |
| `NonStationaryTransformer(NeuralNetworkArchitecture<>,String,NonStationaryTransformerOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a new Non-stationary Transformer instance for ONNX inference mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsChannelIndependent` | Gets whether the model processes channels independently. |
| `NumFeatures` | Gets the number of input features. |
| `PatchSize` | Gets the patch size. |
| `PredictionHorizon` | Gets the prediction horizon. |
| `SequenceLength` | Gets the input sequence length. |
| `Stride` | Gets the stride. |
| `SupportsTraining` | Gets whether this network supports training. |
| `UseNativeMode` | Gets whether the model uses native mode (true) or ONNX mode (false). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyInstanceNormalization(Tensor<>)` | Applies instance normalization (RevIN) to the input. |
| `ApplySeriesStationarization(Tensor<>,Boolean)` | Applies Series Stationarization or its inverse (de-stationarization). |
| `AutoregressiveForecast(Tensor<>,Int32)` | Generates multi-step forecasts using autoregressive prediction. |
| `ConvertFromOnnxTensor(Tensor<Single>)` | Converts an ONNX tensor back to our tensor type. |
| `ConvertToFloatArray(Tensor<>)` | Converts our tensor to a float array for ONNX. |
| `CreateNewInstance` | Creates a new instance of this network type. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes network-specific data from persistence. |
| `Dispose(Boolean)` | Disposes of managed resources. |
| `Evaluate(Tensor<>,Tensor<>)` | Evaluates the model on a test dataset. |
| `ExtractLayerReferences` | Extracts references to specific layers for the Non-stationary Transformer architecture. |
| `Forecast(Tensor<>,Double[])` | Generates forecasts for the given historical data. |
| `ForecastNative(Tensor<>,Int32)` | Performs native forecasting using the built-in layers. |
| `ForecastOnnx(Tensor<>)` | Performs ONNX-based forecasting using the loaded model. |
| `Forward(Tensor<>)` | Performs the forward pass through the network. |
| `ForwardForTraining(Tensor<>)` | Tape-aware training forward. |
| `GetFinancialMetrics` | Gets financial metrics specific to the model. |
| `GetModelMetadata` | Gets metadata about the model. |
| `GetOptions` |  |
| `InitializeDeStationaryParameters` | Initializes the learnable de-stationarization parameters (tau and delta). |
| `InitializeLayers` | Initializes the neural network layers. |
| `PredictCore(Tensor<>)` | Performs prediction on the given input. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes network-specific data for persistence. |
| `ShiftAndAppend(Tensor<>,Tensor<>)` | Shifts the input window and appends new predictions for autoregressive forecasting. |
| `Train(Tensor<>,Tensor<>)` | Trains the network on a single batch. |
| `UpdateParameters(Vector<>)` | Updates network parameters based on gradients. |
| `ValidateCustomLayers(List<ILayer<>>)` | Validates custom layers provided through the architecture. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_deStatProjection` | De-stationarization projection layer. |
| `_decoderLayers` | Decoder layers for generating predictions. |
| `_delta` | Learned delta parameter for De-stationary Attention. |
| `_dropout` | Dropout rate. |
| `_encoderLayers` | Encoder layers for processing input sequence. |
| `_feedForwardDim` | Feedforward dimension. |
| `_inputProjection` | Input projection layer. |
| `_instanceMean` | Instance mean for Series Stationarization. |
| `_instanceStd` | Instance standard deviation for Series Stationarization. |
| `_labelLength` | The label length for decoder start position. |
| `_lossFunction` | The loss function for training. |
| `_modelDimension` | Model dimension (embedding size). |
| `_numDecoderLayers` | Number of decoder layers. |
| `_numEncoderLayers` | Number of encoder layers. |
| `_numFeatures` | Number of input features. |
| `_numHeads` | Number of attention heads. |
| `_optimizer` | The optimizer for training. |
| `_outputProjection` | Output projection layer. |
| `_predictionHorizon` | The prediction horizon. |
| `_projectionDim` | Projection dimension for de-stationarization. |
| `_sequenceLength` | The input sequence length (lookback window). |
| `_tau` | Learned tau parameter for De-stationary Attention. |
| `_useDeStationaryAttention` | Whether to use De-stationary Attention. |
| `_useNativeMode` | Indicates whether this network uses native layers (true) or ONNX model (false). |
| `_useSeriesStationarization` | Whether to use Series Stationarization. |

