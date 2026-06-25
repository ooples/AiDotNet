---
title: "TimesFM<T>"
description: "TimesFM (Time Series Foundation Model) for zero-shot time series forecasting."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Forecasting.Foundation`

TimesFM (Time Series Foundation Model) for zero-shot time series forecasting.

## For Beginners

TimesFM is a revolutionary approach to time series forecasting:

**Foundation Model Concept:**
Just like GPT learns language patterns from vast text data, TimesFM learns time series
patterns from millions of diverse series:

- Weather data (temperature, precipitation, wind)
- Financial data (stock prices, exchange rates)
- Retail data (sales, inventory, demand)
- Energy data (consumption, production, prices)

**Zero-Shot Capability:**
The "zero-shot" term means no training required for new tasks:

- Traditional models: Train specifically for your data
- TimesFM: Works immediately on any time series
- Just provide history → get forecasts

**Patching Innovation:**
TimesFM groups consecutive time steps into "patches":

- Example: 512 time steps → 16 patches of 32 steps each
- Each patch becomes one token for the transformer
- Benefits: Longer context, faster processing, captures local patterns

**Decoder-Only Architecture:**
Like GPT, TimesFM uses causal (one-directional) attention:

- Each position only attends to earlier positions
- Naturally suited for autoregressive forecasting
- Generates predictions step by step

**When TimesFM Excels:**

- Quick prototyping without model training
- Cross-domain forecasting (same model for different data types)
- Limited historical data (leverages pre-training knowledge)
- Baseline comparisons for specialized models

## How It Works

TimesFM is Google's foundation model for time series forecasting. It uses a decoder-only
transformer architecture pre-trained on a massive dataset spanning diverse domains, enabling
zero-shot forecasting without task-specific training.

**Reference:** Das et al., "A decoder-only foundation model for time-series forecasting", 2024.
https://arxiv.org/abs/2310.10688

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TimesFM(NeuralNetworkArchitecture<>,String,TimesFMOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a TimesFM model using pretrained ONNX model. |
| `TimesFM(NeuralNetworkArchitecture<>,TimesFMOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a TimesFM model in native mode for training or fine-tuning. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsChannelIndependent` |  |
| `MaxContextLength` |  |
| `MaxPredictionHorizon` |  |
| `ModelSize` |  |
| `NumFeatures` |  |
| `NumHeads` | Gets the number of attention heads in the transformer. |
| `NumPatches` | Gets the number of patches derived from the context length and patch size. |
| `PatchSize` |  |
| `PredictionHorizon` |  |
| `SequenceLength` |  |
| `Stride` |  |
| `SupportsTraining` |  |
| `UseNativeMode` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddTensors(Tensor<>,Tensor<>)` | Element-wise addition of two tensors. |
| `ApplyInstanceNormalization(Tensor<>)` |  |
| `AutoregressiveForecast(Tensor<>,Int32)` |  |
| `ConcatenatePredictions(List<Tensor<>>,Int32)` | Concatenates multiple prediction tensors for extended horizons. |
| `CreateNewInstance` | Creates a new instance of this model with the same configuration. |
| `CreateQuantileConditionedInput(Tensor<>,Double)` | Creates a quantile-conditioned input by modulating hidden states with the quantile level. |
| `DenormalizeForecast(Tensor<>)` | RevIN reverse step: restores each instance's mean/std to the forecast so it is expressed on the input's original scale (Kim et al. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Reads TimesFM-specific configuration during deserialization. |
| `Dispose(Boolean)` | Disposes resources used by the TimesFM model. |
| `Evaluate(Tensor<>,Tensor<>)` |  |
| `ExtractLayerReferences` | Extracts references to specific layers from the layer collection. |
| `Forecast(Tensor<>,Double[])` |  |
| `ForecastNative(Tensor<>)` | Performs native mode forecasting. |
| `ForecastOnnx(Tensor<>)` | Performs ONNX mode forecasting. |
| `Forward(Tensor<>)` | Performs the forward pass through TimesFM. |
| `ForwardNativeForTraining(Tensor<>)` | TimesFM training-mode forward. |
| `GetFinancialMetrics` |  |
| `GetModelMetadata` |  |
| `GetNamedLayerActivations(Tensor<>)` |  |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the neural network layers for TimesFM. |
| `PredictCore(Tensor<>)` |  |
| `ProduceQuantileForecasts(Tensor<>,Double[])` | Produces quantile forecasts using the TimesFM 2.5 continuous quantile head. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Writes TimesFM-specific configuration during serialization. |
| `ShiftInputWithPredictions(Tensor<>,Tensor<>,Int32)` | Shifts input tensor by incorporating predictions for autoregressive forecasting. |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |
| `ValidateCustomLayers(List<ILayer<>>)` | Validates that custom layers meet TimesFM architectural requirements. |
| `ValidateOptions(TimesFMOptions<>)` | Validates the TimesFM options. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_finalLayerNorm` | Layer normalization applied after all transformer layers. |
| `_outputProjection` | Output projection layer that maps hidden states to forecasts. |
| `_patchEmbedding` | Patch embedding layer that converts raw patches to hidden representations. |
| `_positionEmbedding` | Positional embedding layer for encoding patch positions. |
| `_quantileHidden` | Quantile head hidden layer (TimesFM 2.5) that maps hidden states to quantile dimension. |
| `_quantileOutput` | Quantile head output layer (TimesFM 2.5) that produces per-quantile forecasts. |
| `_transformerLayers` | Transformer decoder layers for processing the embedded sequence. |
| `_useNativeMode` | Indicates whether this network uses native layers (true) or ONNX model (false). |

