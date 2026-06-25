---
title: "DeepState<T>"
description: "DeepState (Deep State Space Model) for probabilistic time series forecasting."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Forecasting.Neural`

DeepState (Deep State Space Model) for probabilistic time series forecasting.

## For Beginners

DeepState is like having a statistical model that can learn:

**State Space Model Basics:**
SSMs assume your observations come from hidden "states" that evolve over time:

- z_t = F * z_{t-1} + process_noise (state transition)
- y_t = H * z_t + observation_noise (observation)

The states might represent:

- Level: The current baseline value
- Trend: The direction and rate of change
- Seasonality: Repeating patterns (daily, weekly, yearly)

**Why "Deep" State Space?**
Classical SSMs require manual specification of model structure.
DeepState uses neural networks to:

1. Process historical data with an RNN encoder
2. Generate SSM parameters (F, H matrices) from learned representations
3. Run the SSM forward to produce forecasts

**Advantages:**

- Natural decomposition into interpretable components
- Built-in uncertainty quantification
- Handles multiple time series with shared patterns
- Robust to missing data

**Example:**
For energy demand forecasting:

- The RNN learns that hot weather increases cooling demand
- The SSM captures daily patterns (peak at 6pm) and weekly patterns (less on weekends)
- Forecasts include uncertainty (wider intervals during unusual weather)

## How It Works

DeepState combines the interpretability of classical state space models (SSM) with the
flexibility of deep learning. An RNN encoder learns to parameterize an SSM, which then
produces forecasts with natural uncertainty quantification.

**Reference:** Rangapuram et al., "Deep State Space Models for Time Series Forecasting", 2018.
https://papers.nips.cc/paper/2018/hash/5cf68969fb67aa6082363a6d4e6468e2-Abstract.html

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DeepState(NeuralNetworkArchitecture<>,DeepStateOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a DeepState model in native mode for training from scratch. |
| `DeepState(NeuralNetworkArchitecture<>,String,DeepStateOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a DeepState model using pretrained ONNX model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsChannelIndependent` |  |
| `NumFeatures` |  |
| `PatchSize` |  |
| `PredictionHorizon` |  |
| `SequenceLength` |  |
| `StateDimension` | Gets the state dimension of the SSM. |
| `Stride` |  |
| `SupportsTraining` |  |
| `UseNativeMode` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddTensors(Tensor<>,Tensor<>)` | Adds two tensors element-wise. |
| `ApplyInstanceNormalization(Tensor<>)` |  |
| `AutoregressiveForecast(Tensor<>,Int32)` |  |
| `ConcatenatePredictions(List<Tensor<>>,Int32)` | Concatenates multiple prediction tensors for extended horizons. |
| `CreateNewInstance` | Creates a new instance of this model with the same configuration. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Reads DeepState-specific configuration during deserialization. |
| `Dispose(Boolean)` | Disposes resources used by the DeepState model. |
| `Evaluate(Tensor<>,Tensor<>)` |  |
| `ExtractLayerReferences` | Extracts references to specific layers from the layer collection. |
| `Forecast(Tensor<>,Double[])` |  |
| `ForecastNative(Tensor<>)` | Performs native mode forecasting. |
| `ForecastOnnx(Tensor<>)` | Performs ONNX mode forecasting. |
| `Forward(Tensor<>)` | Performs the forward pass through DeepState. |
| `ForwardNativeForTraining(Tensor<>)` | Training-mode forward: calls `Tensor{` directly so the tape forward stays in training mode. |
| `GetFinancialMetrics` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the neural network layers for DeepState. |
| `PredictCore(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` | Writes DeepState-specific configuration during serialization. |
| `ShiftInputWithPredictions(Tensor<>,Tensor<>,Int32)` | Shifts input tensor by incorporating predictions for autoregressive forecasting. |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |
| `ValidateCustomLayers(List<ILayer<>>)` | Validates that custom layers meet DeepState architectural requirements. |
| `ValidateOptions(DeepStateOptions<>)` | Validates the DeepState options. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_initialStateLayer` | Layer that generates the initial state. |
| `_inputProjection` | Input projection layer. |
| `_observationLayer` | Layer that generates observation matrix parameters (H). |
| `_outputLayer` | Output projection layer. |
| `_rnnLayers` | RNN encoder layers for processing historical sequence. |
| `_stateEvolutionLayer` | Layer for state evolution across forecast horizon. |
| `_transitionLayer` | Layer that generates state transition matrix parameters (F). |
| `_useNativeMode` | Indicates whether this network uses native layers (true) or ONNX model (false). |

