---
title: "Mamba<T>"
description: "Mamba (Selective State Space Model) implementation for time series forecasting."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Forecasting.StateSpace`

Mamba (Selective State Space Model) implementation for time series forecasting.

## For Beginners

Mamba is a breakthrough in efficient sequence modeling:

**The Key Insight:**
Transformers have O(n^2) complexity due to attention, which is slow for long sequences.
State space models (SSMs) have O(n) complexity but are less expressive.
Mamba makes SSM parameters input-dependent (selective), combining the best of both.

**How It Works:**

1. **State Space Model:** Maintains a hidden state updated recurrently
2. **Selective Mechanism:** Parameters (A, B, C, delta) vary with input
3. **Hardware-aware Algorithm:** Efficient implementation via parallel scan
4. **Linear Complexity:** O(n) time and memory for sequence length n

**Advantages:**

- Linear time complexity (vs O(n^2) for attention)
- Handles very long sequences efficiently
- Strong performance on language, audio, and time series
- Hardware-efficient implementation

## How It Works

Mamba is a selective state space model that achieves linear-time complexity for
sequence modeling while maintaining the expressiveness of transformers through
input-dependent (selective) state space parameters.

**Reference:** Gu and Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", 2024.
https://arxiv.org/abs/2312.00752

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Mamba(NeuralNetworkArchitecture<>,MambaOptions<>,Int32,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new instance in native mode for training. |
| `Mamba(NeuralNetworkArchitecture<>,String,MambaOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new instance using an ONNX pretrained model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ExpandFactor` | Gets the expansion factor. |
| `IsChannelIndependent` |  |
| `NumFeatures` |  |
| `PatchSize` |  |
| `PredictionHorizon` |  |
| `SequenceLength` |  |
| `StateDimension` | Gets the state dimension of the SSM. |
| `Stride` |  |
| `SupportsTraining` |  |
| `UseBidirectional` | Gets whether bidirectional processing is used. |
| `UseNativeMode` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyInstanceNormalization(Tensor<>)` |  |
| `AutoregressiveForecast(Tensor<>,Int32)` |  |
| `ConcatenatePredictions(List<Tensor<>>,Int32)` | Concatenates multiple prediction tensors into a single tensor. |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` | Reads Mamba-specific configuration during deserialization. |
| `Dispose(Boolean)` | Releases resources used by the model. |
| `Evaluate(Tensor<>,Tensor<>)` |  |
| `ExtractLayerReferences` | Extracts references to key layers for efficient access. |
| `Forecast(Tensor<>,Double[])` |  |
| `ForecastOnnx(Tensor<>)` | Performs ONNX-based inference for forecasting. |
| `Forward(Tensor<>)` | Performs the forward pass through the network. |
| `GenerateQuantilePredictions(Tensor<>,Double[])` | Generates quantile predictions through dropout-based sampling. |
| `GetFinancialMetrics` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the model layers based on configuration. |
| `NormalizeInputTo3D(Tensor<>)` | Normalizes input to 3D [batch, seqLen, numFeatures] regardless of input shape. |
| `PredictCore(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` | Writes Mamba-specific configuration during serialization. |
| `ShiftInputWithPredictions(Tensor<>,Tensor<>,Int32)` | Shifts input tensor by appending predictions. |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |
| `ValidateCustomLayers(List<ILayer<>>)` | Validates that custom layers meet Mamba requirements. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_contextLength` | Context length for the input sequence. |
| `_convKernelSize` | Convolution kernel size. |
| `_dropout` | Dropout rate. |
| `_dtRank` | Delta rank for dt projection. |
| `_expandFactor` | Expansion factor for inner dimension. |
| `_forecastHorizon` | Forecast horizon for predictions. |
| `_inputEmbedding` | Reference to the input embedding layer. |
| `_lastForwardBatchSize` | Stores the batch size from the last forward pass for use in backward. |
| `_lastForwardSeqLen` | Stores the actual sequence length from the last forward pass for use in backward. |
| `_lossFunction` | The loss function used for training. |
| `_mambaBlocks` | References to the Mamba block layers implementing the real selective SSM. |
| `_modelDimension` | Model dimension (d_model). |
| `_numFeatures` | Number of input features. |
| `_numLayers` | Number of Mamba layers. |
| `_optimizer` | The optimizer used for training. |
| `_outputProjectionLayers` | References to the output projection layers (all DenseLayers after input embedding). |
| `_stateDimension` | State dimension for SSM. |
| `_useBidirectional` | Whether to use bidirectional processing. |
| `_useNativeMode` | Indicates whether the model is running in native mode (true) or ONNX mode (false). |

