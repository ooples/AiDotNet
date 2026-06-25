---
title: "TimeMachine<T>"
description: "TimeMachine (Time Series State Space Model) for multi-scale time series forecasting."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Forecasting.StateSpace`

TimeMachine (Time Series State Space Model) for multi-scale time series forecasting.

## For Beginners

TimeMachine is a modern architecture whose key insight is
that "A Time Series is Worth 4 Mambas" - using multiple SSM blocks at different scales:

**The Core Idea:**
Time series data contains patterns at multiple temporal scales:

- High-frequency noise and short-term fluctuations
- Daily, weekly, monthly patterns
- Long-term trends

TimeMachine captures all these by processing the data at multiple scales simultaneously.

**How It Works:**

1. **Temporal Decomposition:** Separates the signal into multiple scales
2. **Multi-Scale SSM:** Each scale has its own Mamba-style SSM blocks
3. **Scale-wise Attention:** Learns which scales are most important
4. **Reconstruction:** Combines multi-scale outputs for final forecast

**Architecture:**

- Input embedding with reversible instance normalization (RevIN)
- 4 parallel SSM branches at different downsampling rates
- Attention-based fusion of scale outputs
- Output projection with de-normalization

**Key Benefits:**

- Linear complexity O(n) from SSM backbone
- Multi-scale captures patterns at all frequencies
- RevIN handles non-stationarity
- State-of-the-art results on long-term forecasting benchmarks

## How It Works

TimeMachine is a state space model specifically designed for time series forecasting
that combines multiple SSM blocks at different temporal scales to capture both
short-term and long-term patterns effectively.

**Reference:** Ahamed et al., "TimeMachine: A Time Series is Worth 4 Mambas for Long-term Forecasting", 2024.
https://arxiv.org/abs/2403.09898

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TimeMachine(NeuralNetworkArchitecture<>,String,TimeMachineOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new instance of the TimeMachine model in ONNX mode for inference. |
| `TimeMachine(NeuralNetworkArchitecture<>,TimeMachineOptions<>,Int32,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new instance of the TimeMachine model in native mode for training. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ContextLength` | Gets the input context length for the model. |
| `ForecastHorizon` | Gets the forecast horizon (number of future steps to predict). |
| `IsChannelIndependent` |  |
| `NumFeatures` |  |
| `NumScales` | Gets the number of temporal scales used. |
| `PatchSize` |  |
| `PredictionHorizon` |  |
| `SequenceLength` |  |
| `Stride` |  |
| `SupportsTraining` | Gets whether the model supports training (native mode only). |
| `UseMultiScaleAttention` | Gets whether multi-scale attention is used for fusion. |
| `UseNativeMode` |  |
| `UseReversibleNormalization` | Gets whether reversible instance normalization is used. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyInstanceNormalization(Tensor<>)` | Applies instance normalization to the input. |
| `ApplyTemporalDecomposition(Tensor<>,Int32)` | Applies temporal decomposition to separate different frequency components. |
| `AutoregressiveForecast(Tensor<>,Int32)` | Performs autoregressive forecasting step by step. |
| `ComputePredictionIntervals(List<Tensor<>>,Double)` | Computes prediction intervals from Monte Carlo samples. |
| `ConcatenatePredictions(List<Tensor<>>)` | Concatenates multiple prediction tensors into a single tensor. |
| `CreateNewInstance` | Creates a new instance of the TimeMachine model with the same configuration. |
| `DenormalizeForecast(Tensor<>)` | RevIN reverse step (Kim et al. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes TimeMachine-specific data when loading a saved model. |
| `Dispose(Boolean)` | Disposes of managed and unmanaged resources. |
| `Evaluate(Tensor<>,Tensor<>)` | Evaluates forecast quality against actual values. |
| `ExtractLayerReferences` | Extracts references to key layers for efficient access. |
| `FlattenInput(Tensor<>)` | Flattens the input tensor for processing through dense layers. |
| `Forecast(Tensor<>,Double[])` | Generates forecasts for the given input time series. |
| `ForecastNative(Tensor<>)` | Performs native mode forecasting through the layer stack. |
| `ForecastOnnx(Tensor<>)` | Performs ONNX mode forecasting using the pretrained model. |
| `ForecastWithIntervals(Tensor<>,Double)` | Generates forecasts with prediction intervals for uncertainty quantification. |
| `Forward(Tensor<>)` | Performs the forward pass through all layers. |
| `ForwardNativeForTraining(Tensor<>)` | Training-mode forward. |
| `GetFinancialMetrics` | Gets financial-specific metrics about the model. |
| `GetModelMetadata` | Gets metadata about the TimeMachine model. |
| `GetNamedLayerActivations(Tensor<>)` |  |
| `GetOptions` |  |
| `InitializeLayers` | Initializes all layers for the TimeMachine model. |
| `PredictCore(Tensor<>)` | Performs forward prediction on the input tensor. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes TimeMachine-specific data for model persistence. |
| `ShiftInputWindow(Tensor<>,Tensor<>)` | Shifts the input window by removing oldest values and appending new prediction. |
| `Train(Tensor<>,Tensor<>)` | Trains the TimeMachine model on a batch of input-target pairs. |
| `UpdateParameters(Vector<>)` | Updates the model parameters using the optimizer (required override). |
| `ValidateCustomLayers(List<ILayer<>>)` | Validates custom layers provided through the architecture. |

