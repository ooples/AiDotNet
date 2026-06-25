---
title: "Mamba2<T>"
description: "Mamba-2 (State Space Duality) implementation for time series forecasting."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Forecasting.StateSpace`

Mamba-2 (State Space Duality) implementation for time series forecasting.

## For Beginners

Mamba-2 is the next generation of the Mamba architecture:

**Key Improvements over Mamba-1:**

1. **SSD Algorithm:** Uses matrix multiply instead of associative scan — much faster on GPUs
2. **Multi-head Structure:** Like multi-head attention, enabling better capacity per parameter
3. **Chunk-wise Processing:** Processes sequences in chunks for better hardware utilization
4. **2-8x Faster Training:** Due to better hardware mapping

**For Time Series:**

- Efficient handling of long historical windows
- Multi-head captures different temporal patterns simultaneously
- Linear complexity enables real-time forecasting on long sequences

## How It Works

Mamba-2 improves upon Mamba by discovering the connection between selective state space models
and structured masked attention (State Space Duality). This enables a more efficient SSD algorithm
using matrix multiplications and multi-head structure, achieving 2-8x faster training.

**Reference:** Dao and Gu, "Transformers are SSMs: Generalized Models and Efficient Algorithms
Through Structured State Space Duality", 2024.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Mamba2(NeuralNetworkArchitecture<>,Mamba2Options<>,Int32,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new instance in native mode for training. |
| `Mamba2(NeuralNetworkArchitecture<>,String,Mamba2Options<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new instance using an ONNX pretrained model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ChunkSize` | Gets the chunk size for SSD computation. |
| `IsChannelIndependent` |  |
| `NumFeatures` |  |
| `NumHeads` | Gets the number of heads for multi-head SSD. |
| `PatchSize` |  |
| `PredictionHorizon` |  |
| `SequenceLength` |  |
| `StateDimension` | Gets the state dimension per head. |
| `Stride` |  |
| `SupportsTraining` |  |
| `UseNativeMode` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyInstanceNormalization(Tensor<>)` |  |
| `AutoregressiveForecast(Tensor<>,Int32)` |  |
| `ConcatenatePredictions(List<Tensor<>>,Int32)` |  |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Dispose(Boolean)` |  |
| `Evaluate(Tensor<>,Tensor<>)` |  |
| `Forecast(Tensor<>,Double[])` |  |
| `ForecastOnnx(Tensor<>)` |  |
| `GetFinancialMetrics` |  |
| `GetModelMetadata` |  |
| `GetNamedLayerActivations(Tensor<>)` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `PredictCore(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `ShiftInputWithPredictions(Tensor<>,Tensor<>,Int32)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |
| `ValidateCustomLayers(List<ILayer<>>)` |  |

