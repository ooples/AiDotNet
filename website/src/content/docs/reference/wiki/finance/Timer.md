---
title: "Timer<T>"
description: "Timer (Generative Pre-Training for Time Series) implementation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Forecasting.Foundation`

Timer (Generative Pre-Training for Time Series) implementation.

## For Beginners

Timer brings GPT-style pre-training to time series:

**The Key Insight:**
Just like GPT learns language by predicting the next token, Timer learns
time series patterns by predicting future values. Pre-training on diverse
datasets enables strong zero-shot transfer.

**How It Works:**

1. **Autoregressive Pre-training:** Learn to predict future from past
2. **Masked Modeling:** Learn to reconstruct masked portions
3. **Multi-scale Processing:** Handle different temporal granularities
4. **Fine-tuning:** Adapt to specific domains with minimal data

**Advantages:**

- Strong zero-shot and few-shot performance
- Generalizes across domains and frequencies
- Efficient fine-tuning with minimal labeled data
- Handles variable sequence lengths

## How It Works

Timer is a generative pre-training approach for time series that uses
autoregressive generation to learn rich temporal representations from
diverse time series datasets, similar to GPT for language.

**Timer-XL Enhancements (2024):**
Timer-XL extends the original Timer with long-context support and a unified
framework for multiple forecasting tasks. Key improvements:

- Extended context length (up to 4096 time steps)
- Unified multi-task forecasting framework
- Improved long-horizon performance

**Reference:** Liu et al., "Timer: Generative Pre-Training of Time Series", 2024.
https://arxiv.org/abs/2402.02368
Timer-XL: "Timer-XL: Long-Context Transformers for Unified Time Series Forecasting", 2024.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Timer(NeuralNetworkArchitecture<>,String,TimerOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new instance using an ONNX pretrained model. |
| `Timer(NeuralNetworkArchitecture<>,TimerOptions<>,Int32,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new instance in native mode for training. |

## Properties

| Property | Summary |
|:-----|:--------|
| `GenerationTemperature` | Gets the generation temperature. |
| `IsChannelIndependent` |  |
| `MaskRatio` | Gets the mask ratio for masked modeling. |
| `MaxContextLength` |  |
| `MaxPredictionHorizon` |  |
| `ModelSize` |  |
| `NumFeatures` |  |
| `PatchSize` |  |
| `PredictionHorizon` |  |
| `SequenceLength` |  |
| `Stride` |  |
| `SupportsTraining` |  |
| `UseAutoregressiveDecoding` | Gets whether autoregressive decoding is used. |
| `UseNativeMode` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyInstanceNormalization(Tensor<>)` |  |
| `AutoregressiveForecast(Tensor<>,Int32)` |  |
| `AutoregressiveGenerate(Tensor<>,Int32)` | Performs autoregressive generation step by step. |
| `ConcatenatePredictions(List<Tensor<>>,Int32)` | Concatenates multiple prediction tensors into a single tensor. |
| `CreateNewInstance` |  |
| `DenormalizeForecast(Tensor<>)` | RevIN reverse step: restores each instance's mean/std to the forecast so it is expressed on the input's original scale (Kim et al. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Reads Timer-specific configuration during deserialization. |
| `Dispose(Boolean)` | Releases resources used by the model. |
| `Evaluate(Tensor<>,Tensor<>)` |  |
| `ExtractLayerReferences` | Extracts references to key layers for efficient access. |
| `Forecast(Tensor<>,Double[])` |  |
| `ForecastOnnx(Tensor<>)` | Performs ONNX-based inference for forecasting. |
| `ForwardNativeForTraining(Tensor<>)` | Timer training-mode forward. |
| `GenerateQuantilePredictions(Tensor<>,Double[])` | Generates quantile predictions through temperature-based sampling. |
| `GenerateRandomMask(Tensor<>)` | Generates a random mask tensor for masked pre-training. |
| `GetFinancialMetrics` |  |
| `GetModelMetadata` |  |
| `GetNamedLayerActivations(Tensor<>)` | Performs the forward pass through the network. |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the model layers based on configuration. |
| `MaskedPretraining(Tensor<>,Tensor<>)` | Performs masked modeling pre-training step. |
| `PredictCore(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` | Writes Timer-specific configuration during serialization. |
| `ShiftInputWithPredictions(Tensor<>,Tensor<>,Int32)` | Shifts input tensor by appending predictions. |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |
| `ValidateCustomLayers(List<ILayer<>>)` | Validates that custom layers meet Timer requirements. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_contextLength` | Context length for the input sequence. |
| `_dropout` | Dropout rate. |
| `_forecastHorizon` | Forecast horizon for predictions. |
| `_generationHead` | Reference to the generation head layer. |
| `_generationTemperature` | Temperature for sampling during generation. |
| `_hiddenDimension` | Hidden dimension size. |
| `_lossFunction` | The loss function used for training. |
| `_maskRatio` | Mask ratio for masked modeling. |
| `_numFeatures` | Number of input features. |
| `_numHeads` | Number of attention heads. |
| `_numLayers` | Number of transformer layers. |
| `_numPatches` | Number of patches. |
| `_optimizer` | The optimizer used for training. |
| `_patchEmbedding` | Reference to the patch embedding layer. |
| `_patchLength` | Patch length for tokenization. |
| `_patchStride` | Patch stride. |
| `_revinMean` | RevIN (reversible instance normalization, Kim et al. |
| `_transformerLayers` | References to the transformer decoder layers. |
| `_useAutoregressiveDecoding` | Whether to use autoregressive decoding. |
| `_useNativeMode` | Indicates whether the model is running in native mode (true) or ONNX mode (false). |

