---
title: "NBEATSFinance<T>"
description: "N-BEATS (Neural Basis Expansion Analysis for Time Series) model for financial forecasting."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Forecasting.Neural`

N-BEATS (Neural Basis Expansion Analysis for Time Series) model for financial forecasting.

## For Beginners

N-BEATS works by stacking multiple "blocks" that each try to explain
part of the time series. Each block:

- Looks at the input
- Produces a "backcast" (its explanation of the past)
- Produces a "forecast" (its prediction for the future)
- Passes the "residual" (what it couldn't explain) to the next block

This hierarchical approach allows N-BEATS to decompose complex patterns into simpler components,
similar to how you might break down a sound wave into different frequency components.

Key benefits:

- **Interpretable:** Can separate trend from seasonality
- **No feature engineering:** Works directly on raw time series
- **State-of-the-art accuracy:** Competitive with or better than traditional methods

## How It Works

N-BEATS is a deep neural architecture that uses basis expansion to decompose time series
into interpretable components. It achieves state-of-the-art performance while providing
the ability to decompose forecasts into trend and seasonality components.

**Reference:** Oreshkin et al., "N-BEATS: Neural basis expansion analysis for
interpretable time series forecasting", ICLR 2020. https://arxiv.org/abs/1905.10437

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NBEATSFinance(NeuralNetworkArchitecture<>,NBEATSModelOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates an N-BEATS network in native mode for training from scratch. |
| `NBEATSFinance(NeuralNetworkArchitecture<>,String,NBEATSModelOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates an N-BEATS network using pretrained ONNX model. |

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
| `ApplyInstanceNormalization(Tensor<>)` |  |
| `AutoregressiveForecast(Tensor<>,Int32)` |  |
| `ConcatenatePredictions(List<Tensor<>>,Int32)` | Concatenates multiple predictions into a single tensor. |
| `CreateNewInstance` | Creates a new instance of this model with the same configuration. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Reads N-BEATS-specific configuration during deserialization. |
| `Dispose(Boolean)` | Releases resources used by the N-BEATS model. |
| `Evaluate(Tensor<>,Tensor<>)` |  |
| `ExtractLayerReferences` | Extracts references to specific layers from the layer collection. |
| `Forecast(Tensor<>,Double[])` |  |
| `ForecastNative(Tensor<>)` | Performs native mode forecasting. |
| `ForecastOnnx(Tensor<>)` | Performs ONNX mode forecasting. |
| `Forward(Tensor<>)` | Performs the forward pass through the N-BEATS network. |
| `ForwardNativeForTraining(Tensor<>)` | Training-mode forward: calls `Tensor{` directly so block-level dropout in N-BEATS fully-connected stacks stays active during backprop. |
| `GetFinancialMetrics` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the neural network layers for N-BEATS. |
| `PredictCore(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` | Writes N-BEATS-specific configuration during serialization. |
| `ShiftInputWithPredictions(Tensor<>,Tensor<>,Int32)` | Shifts input by incorporating recent predictions. |
| `SubtractTensors(Tensor<>,Tensor<>)` | Subtracts two tensors element-wise. |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |
| `ValidateCustomLayers(List<ILayer<>>)` | Validates that custom layers meet N-BEATS architectural requirements. |
| `ValidateOptions(NBEATSModelOptions<>)` | Validates the N-BEATS options. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_blocks` | Groups of layers representing each N-BEATS block. |
| `_cachedBlockHiddenOutputs` | Cached per-block hidden layer outputs from Forward pass for proper gradient routing. |
| `_forecastHorizon` | The forecast horizon. |
| `_hiddenSize` | Size of hidden layers. |
| `_lookbackWindow` | The lookback window size. |
| `_lossFunction` | The loss function for training. |
| `_numBlocksPerStack` | Number of blocks per stack. |
| `_numHiddenLayers` | Number of hidden layers per block. |
| `_numStacks` | Number of stacks. |
| `_optimizer` | The optimizer for training. |
| `_outputProjection` | Final output projection layer. |
| `_polynomialDegree` | Polynomial degree for trend basis. |
| `_shareWeightsInStack` | Whether to share weights within stacks. |
| `_useInterpretableBasis` | Whether to use interpretable basis functions. |
| `_useNativeMode` | Indicates whether this network uses native layers (true) or ONNX model (false). |

