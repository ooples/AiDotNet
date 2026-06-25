---
title: "LSTNet<T>"
description: "LSTNet (Long Short-Term Time-series Network) model for multivariate time series forecasting."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Forecasting.Neural`

LSTNet (Long Short-Term Time-series Network) model for multivariate time series forecasting.

## For Beginners

LSTNet is like having a team of specialists analyze your time series:

1. **The Pattern Scanner (CNN)**: Scans through your data looking for local patterns,

like finding "every Monday is busy" or "there's always a dip at noon".

2. **The Trend Tracker (GRU)**: Remembers long-term trends, like "sales have been

growing steadily for the past 3 months".

3. **The Season Watcher (Skip-RNN)**: Compares the same time across different periods,

like comparing this Tuesday's 3 PM with last Tuesday's 3 PM.

4. **The Simple Forecaster (Autoregressive)**: Makes basic predictions based on recent

values, like "if it was 100 yesterday, it's probably around 100 today".

All these specialists combine their insights to produce the final forecast.

## How It Works

LSTNet combines multiple neural network components to capture patterns at different temporal scales:

- Convolutional layers for short-term local patterns
- Recurrent layers (GRU) for long-term dependencies
- Skip-RNN for periodic patterns
- Autoregressive component for simple linear relationships

**Reference:** Lai et al., "Modeling Long- and Short-Term Temporal Patterns with Deep
Neural Networks", SIGIR 2018. https://arxiv.org/abs/1703.07015

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LSTNet(NeuralNetworkArchitecture<>,LSTNetOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates an LSTNet in native mode for training from scratch. |
| `LSTNet(NeuralNetworkArchitecture<>,String,LSTNetOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates an LSTNet using pretrained ONNX model. |

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
| `AutoregressiveForecast(Tensor<>,Int32)` |  |
| `CombineOutputs(Tensor<>,Tensor<>)` | Combines outputs from GRU, Skip-GRU, and Highway components. |
| `ConcatenatePredictions(List<Tensor<>>,Int32)` | Concatenates multiple prediction tensors into a single result. |
| `CreateNewInstance` | Creates a new instance of this model with the same configuration. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Reads LSTNet-specific configuration during deserialization. |
| `Dispose(Boolean)` | Releases managed resources used by this model. |
| `Evaluate(Tensor<>,Tensor<>)` |  |
| `ExtractLayerReferences` | Extracts references to specific layers from the layer collection. |
| `ExtractRecentValues(Tensor<>,Int32)` | Extracts the most recent values from the input for autoregressive processing. |
| `Forecast(Tensor<>,Double[])` |  |
| `ForecastNative(Tensor<>)` | Performs native mode forecasting. |
| `ForecastOnnx(Tensor<>)` | Performs ONNX mode forecasting. |
| `Forward(Tensor<>)` | Performs the forward pass through the LSTNet. |
| `GetFinancialMetrics` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the neural network layers for LSTNet. |
| `PredictCore(Tensor<>)` |  |
| `SampleAtSkipIntervals(Tensor<>,Int32)` | Samples the input at skip intervals to capture periodic patterns. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Writes LSTNet-specific configuration during serialization. |
| `ShiftInputWithPredictions(Tensor<>,Tensor<>,Int32)` | Shifts the input tensor by incorporating new predictions. |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |
| `ValidateCustomLayers(List<ILayer<>>)` | Validates that custom layers meet LSTNet architectural requirements. |
| `ValidateOptions(LSTNetOptions<>)` | Validates the LSTNet options. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_autoregressiveWindow` | Window size for autoregressive component. |
| `_combinationLayer` | Layer to combine outputs from different components. |
| `_convActivation` | Activation layer after convolution. |
| `_convDropout` | Dropout layer after convolution for regularization. |
| `_convLayer` | The convolutional layer for extracting local features. |
| `_convolutionFilters` | Number of convolutional filters. |
| `_convolutionKernelSize` | Kernel size for convolutional layer. |
| `_dropout` | Dropout rate for regularization. |
| `_forecastHorizon` | The forecast horizon. |
| `_gruDropout` | Dropout layer after GRU. |
| `_gruLayer` | The main recurrent layer (GRU) for long-term dependencies. |
| `_hiddenRecurrentSize` | Hidden size for main recurrent layer. |
| `_hiddenSkipSize` | Hidden size for skip recurrent layer. |
| `_highwayLayer` | Highway layer for linear pass-through. |
| `_lookbackWindow` | The lookback window size. |
| `_lossFunction` | The loss function for training. |
| `_numFeatures` | Number of input features. |
| `_optimizer` | The optimizer for training. |
| `_outputLayer` | Final output projection layer. |
| `_skipDropout` | Dropout layer after skip GRU. |
| `_skipGruLayer` | The skip recurrent layer for periodic patterns. |
| `_skipPeriod` | Skip period for skip-RNN. |
| `_useHighway` | Whether to use highway connections. |
| `_useNativeMode` | Indicates whether this network uses native layers (true) or ONNX model (false). |

