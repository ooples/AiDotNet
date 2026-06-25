---
title: "TinyTimeMixers<T>"
description: "Tiny Time Mixers (TTM) foundation model for compact, high-performance time series forecasting."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Forecasting.Foundation`

Tiny Time Mixers (TTM) foundation model for compact, high-performance time series forecasting.

## For Beginners

TTM proves that bigger isn't always better:

**MLP-Mixer Architecture:**
Instead of expensive attention mechanisms, TTM alternates between two types of mixing:

1. **Temporal Mixing:** An MLP that mixes information across time steps within each feature.

Think of it as learning "how does the pattern at time t relate to time t-1, t-2, etc.?"

2. **Channel Mixing:** An MLP that mixes information across features at each time step.

Think of it as learning "how does price relate to volume at this moment?"

**Why So Small Yet Effective:**

- Time series have simpler structure than language or images
- MLP-Mixers capture temporal patterns efficiently without attention overhead
- Patch-based input reduces the sequence length the model needs to process
- Careful architectural choices maximize information per parameter

**Practical Benefits:**

- Runs on CPU in real-time (no GPU required)
- Trains in minutes instead of hours
- Perfect for edge deployment (IoT, mobile)
- Low memory footprint (~20MB model size)

## How It Works

TTM is IBM Research's lightweight foundation model that uses an MLP-Mixer architecture
instead of attention-based transformers. With only 1-5 million parameters, it outperforms
or matches models 20-40x its size on standard forecasting benchmarks.

**Reference:** Ekambaram et al., "Tiny Time Mixers (TTMs): Fast Pre-trained Models
for Enhanced Zero/Few-Shot Forecasting of Multivariate Time Series", NeurIPS 2024.
https://arxiv.org/abs/2401.03955

**Thread Safety:** This class is NOT thread-safe. Create separate instances for concurrent usage.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TinyTimeMixers(NeuralNetworkArchitecture<>,String,TinyTimeMixersOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a Tiny Time Mixers model using a pretrained ONNX model. |
| `TinyTimeMixers(NeuralNetworkArchitecture<>,TinyTimeMixersOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a Tiny Time Mixers model in native mode for training or fine-tuning. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsChannelIndependent` |  |
| `MaxContextLength` |  |
| `MaxPredictionHorizon` |  |
| `ModelSize` |  |
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
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Evaluate(Tensor<>,Tensor<>)` |  |
| `Forecast(Tensor<>,Double[])` |  |
| `ForecastOnnx(Tensor<>)` | Runs inference using the ONNX model. |
| `ForwardNative(Tensor<>)` | Performs the full native forward pass through the TTM MLP-Mixer architecture. |
| `GetFinancialMetrics` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `PredictCore(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |
| `ValidateCustomLayers(List<ILayer<>>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_useNativeMode` | Indicates whether this model uses native layers (true) or ONNX model (false). |

