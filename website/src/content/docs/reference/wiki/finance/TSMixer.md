---
title: "TSMixer<T>"
description: "TSMixer: An all-MLP architecture for time series forecasting."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Forecasting.Transformers`

TSMixer: An all-MLP architecture for time series forecasting.

## For Beginners

TSMixer is a simpler alternative to transformer-based models that:

- Uses only MLPs (fully connected layers)
- Alternates between mixing information across time and across features
- Is faster to train and more memory-efficient than attention-based models

The core idea is that mixing time and feature information separately but alternately
is sufficient for capturing complex patterns in multivariate time series.

## How It Works

TSMixer achieves state-of-the-art results using only multilayer perceptrons (MLPs),
without attention mechanisms. It alternates between time-mixing and feature-mixing
operations to capture both temporal patterns and cross-variable relationships.

**Reference:** Chen et al., "TSMixer: An All-MLP Architecture for Time Series
Forecasting", TMLR 2023. https://arxiv.org/abs/2303.06053

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TSMixer(NeuralNetworkArchitecture<>,String,TSMixerOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a new TSMixer instance for ONNX inference mode. |
| `TSMixer(NeuralNetworkArchitecture<>,TSMixerOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a new TSMixer instance for native training mode. |

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
| `ApplyRevIN(Tensor<>,Boolean)` | Applies RevIN (Reversible Instance Normalization). |
| `AutoregressiveForecast(Tensor<>,Int32)` | Generates multi-step forecasts using autoregressive prediction. |
| `ConvertFromOnnxTensor(Tensor<Single>)` | Converts an ONNX tensor back to our tensor type. |
| `ConvertToFloatArray(Tensor<>)` | Converts our tensor to a float array for ONNX. |
| `CreateNewInstance` | Creates a new instance of this network type. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes network-specific data from persistence. |
| `Dispose(Boolean)` | Disposes of managed resources. |
| `Evaluate(Tensor<>,Tensor<>)` | Evaluates the model on a test dataset. |
| `ExtractLayerReferences` | Extracts references to specific layers for the TSMixer architecture. |
| `Forecast(Tensor<>,Double[])` | Generates forecasts for the given historical data. |
| `ForecastNative(Tensor<>,Int32)` | Performs native forecasting using the built-in layers. |
| `ForecastOnnx(Tensor<>)` | Performs ONNX-based forecasting using the loaded model. |
| `Forward(Tensor<>)` | Performs the forward pass through the network. |
| `ForwardForTraining(Tensor<>)` | Tape-aware training forward. |
| `GetFinancialMetrics` | Gets financial metrics specific to the model. |
| `GetModelMetadata` | Gets metadata about the model. |
| `GetOptions` |  |
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
| `_dropout` | Dropout rate. |
| `_featuresFirst` | Whether to mix features before time. |
| `_feedForwardExpansion` | Feedforward expansion factor. |
| `_hiddenDim` | Hidden dimension for MLP layers. |
| `_inputProjection` | Input projection layer. |
| `_instanceMean` | Instance mean for RevIN normalization. |
| `_instanceStd` | Instance standard deviation for RevIN normalization. |
| `_lossFunction` | The loss function for training. |
| `_mixerBlocks` | Mixer blocks containing time-mixing and feature-mixing MLPs. |
| `_numBlocks` | Number of mixer blocks. |
| `_numFeatures` | Number of input features. |
| `_optimizer` | The optimizer for training. |
| `_outputProjection` | Output projection layer. |
| `_predictionHorizon` | The prediction horizon. |
| `_sequenceLength` | The input sequence length (lookback window). |
| `_temporalProjection` | Temporal projection layer. |
| `_useNativeMode` | Indicates whether this network uses native layers (true) or ONNX model (false). |
| `_useRevIN` | Whether to use RevIN normalization. |

