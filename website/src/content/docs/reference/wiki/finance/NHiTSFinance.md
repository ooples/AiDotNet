---
title: "NHiTSFinance<T>"
description: "N-HiTS (Neural Hierarchical Interpolation for Time Series) model for financial forecasting."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Forecasting.Neural`

N-HiTS (Neural Hierarchical Interpolation for Time Series) model for financial forecasting.

## For Beginners

N-HiTS works like having multiple "zoom levels" on your time series:

- One level looks at fine details (hourly patterns)
- Another looks at medium patterns (daily patterns)
- Another looks at the big picture (weekly/monthly trends)

Each level (stack) samples the data at different rates:

- High-frequency stack: Processes data at full resolution
- Medium-frequency stack: Downsamples by 4x (averages 4 points into 1)
- Low-frequency stack: Downsamples by 8x

This multi-scale approach helps N-HiTS:

- Be more efficient (fewer parameters than N-BEATS)
- Handle long horizons better
- Capture patterns at different time scales

## How It Works

N-HiTS improves upon N-BEATS by incorporating hierarchical interpolation and multi-rate
signal sampling. It achieves better accuracy on long-horizon forecasting while being
more parameter-efficient through its stack-specific pooling approach.

**Reference:** Challu et al., "N-HiTS: Neural Hierarchical Interpolation for Time Series
Forecasting", AAAI 2023. https://arxiv.org/abs/2201.12886

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NHiTSFinance(NeuralNetworkArchitecture<>,NHiTSOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates an N-HiTS network in native mode for training from scratch. |
| `NHiTSFinance(NeuralNetworkArchitecture<>,String,NHiTSOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates an N-HiTS network using pretrained ONNX model. |

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
| `ApplyPooling(Tensor<>,Int32)` | Applies pooling to reduce input resolution. |
| `ApplyPoolingTape(Tensor<>,Int32)` | Tape-aware average pooling. |
| `AutoregressiveForecast(Tensor<>,Int32)` |  |
| `ConcatenatePredictions(List<Tensor<>>,Int32)` | Concatenates multiple predictions into a single tensor. |
| `CreateNewInstance` | Creates a new instance of this model with the same configuration. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Reads N-HiTS-specific configuration during deserialization. |
| `Dispose(Boolean)` | Releases resources used by the N-HiTS model. |
| `Evaluate(Tensor<>,Tensor<>)` |  |
| `ExtractLayerReferences` | Extracts references to specific layers from the layer collection. |
| `Forecast(Tensor<>,Double[])` |  |
| `ForecastNative(Tensor<>)` | Performs native mode forecasting. |
| `ForecastOnnx(Tensor<>)` | Performs ONNX mode forecasting. |
| `Forward(Tensor<>)` | Performs the forward pass through the N-HiTS network. |
| `ForwardNativeForTraining(Tensor<>)` | Training-mode forward: replays the N-HiTS pipeline (pool → block → coefficients → interpolate → residual subtract/forecast add) but through `Int32)` and `Int32)`, which use `Engine.ReduceMean` / `Engine.TensorMatMul` instead of the tape-bre… |
| `GetFinancialMetrics` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the neural network layers for N-HiTS. |
| `InterpolateToLength(Tensor<>,Int32)` | Interpolates coefficients to target length. |
| `InterpolateToLengthTape(Tensor<>,Int32)` | Tape-aware linear interpolation from `coeffs` of length `coeffLen` to a target length. |
| `PredictCore(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` | Writes N-HiTS-specific configuration during serialization. |
| `ShiftInputWithPredictions(Tensor<>,Tensor<>,Int32)` | Shifts input by incorporating recent predictions. |
| `SubtractTensors(Tensor<>,Tensor<>)` | Subtracts two tensors element-wise. |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |
| `ValidateCustomLayers(List<ILayer<>>)` | Validates that custom layers meet N-HiTS architectural requirements. |
| `ValidateOptions(NHiTSOptions<>)` | Validates the N-HiTS options. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_dropout` | Dropout rate for regularization. |
| `_forecastHorizon` | The forecast horizon. |
| `_hiddenSize` | Size of hidden layers. |
| `_interpolationModes` | Interpolation modes for each stack. |
| `_lookbackWindow` | The lookback window size. |
| `_lossFunction` | The loss function for training. |
| `_numBlocksPerStack` | Number of blocks per stack. |
| `_numHiddenLayers` | Number of hidden layers per block. |
| `_numStacks` | Number of stacks. |
| `_optimizer` | The optimizer for training. |
| `_outputProjection` | Final output projection layer. |
| `_poolingKernelSizes` | Pooling kernel sizes for each stack. |
| `_poolingModes` | Pooling modes for each stack. |
| `_stackBlocks` | Groups of layers for each stack, organized by resolution level. |
| `_useNativeMode` | Indicates whether this network uses native layers (true) or ONNX model (false). |

