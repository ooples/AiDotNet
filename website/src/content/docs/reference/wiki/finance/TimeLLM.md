---
title: "TimeLLM<T>"
description: "Time-LLM (Large Language Model Reprogramming for Time Series) implementation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Forecasting.Foundation`

Time-LLM (Large Language Model Reprogramming for Time Series) implementation.

## For Beginners

Time-LLM is a clever way to use powerful language models for time series:

**The Key Insight:**
LLMs like GPT/LLaMA are amazing at pattern recognition in sequences.
Time-LLM asks: "Can we make time series 'speak' the language of LLMs?"

**How It Works:**

1. **Patch Reprogramming:** Convert time series patches into "prompt-like" tokens
2. **Text Prototypes:** Learn embeddings that bridge numeric and text domains
3. **Frozen LLM:** The LLM weights stay fixed (no fine-tuning needed)
4. **Output Projection:** Map LLM output back to forecast values

**Advantages:**

- Leverages powerful pretrained LLMs without expensive fine-tuning
- Works with any LLM backbone (GPT-2, LLaMA, etc.)
- Only trains small reprogramming layers
- Zero-shot transfer to new domains

## How It Works

Time-LLM repurposes frozen large language models for time series forecasting by
learning a reprogramming layer that translates time series into text-like representations
that the LLM can understand.

**Reference:** Jin et al., "Time-LLM: Time Series Forecasting by Reprogramming Large Language Models", 2024.
https://arxiv.org/abs/2310.01728

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TimeLLM(NeuralNetworkArchitecture<>,String,TimeLLMOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new instance using an ONNX pretrained model. |
| `TimeLLM(NeuralNetworkArchitecture<>,TimeLLMOptions<>,Int32,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new instance in native mode for training. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsChannelIndependent` |  |
| `LLMBackbone` | Gets the LLM backbone type. |
| `NumFeatures` |  |
| `NumPrototypes` | Gets the number of text prototypes. |
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
| `ConcatenatePredictions(List<Tensor<>>,Int32)` | Concatenates multiple prediction tensors into a single tensor. |
| `CreateNewInstance` |  |
| `DenormalizeForecast(Tensor<>)` | RevIN reverse step: restores each instance's mean/std to the forecast so it is expressed on the input's original scale (Kim et al. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Reads Time-LLM-specific configuration during deserialization. |
| `Dispose(Boolean)` | Releases resources used by the model. |
| `Evaluate(Tensor<>,Tensor<>)` |  |
| `ExtractLayerReferences` | Extracts references to key layers for efficient access. |
| `Forecast(Tensor<>,Double[])` |  |
| `ForecastOnnx(Tensor<>)` | Performs ONNX-based inference for forecasting. |
| `Forward(Tensor<>)` | Performs the forward pass through the network. |
| `ForwardNativeForTraining(Tensor<>)` | Time-LLM training-mode forward. |
| `GenerateQuantilePredictions(Tensor<>,Double[])` | Generates quantile predictions through Monte Carlo dropout. |
| `GetFinancialMetrics` |  |
| `GetModelMetadata` |  |
| `GetNamedLayerActivations(Tensor<>)` |  |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the model layers based on configuration. |
| `PredictCore(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` | Writes Time-LLM-specific configuration during serialization. |
| `ShiftInputWithPredictions(Tensor<>,Tensor<>,Int32)` | Shifts input tensor by appending predictions. |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |
| `ValidateCustomLayers(List<ILayer<>>)` | Validates that custom layers meet Time-LLM requirements. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_contextLength` | Context length for the input sequence. |
| `_dropout` | Dropout rate. |
| `_forecastHorizon` | Forecast horizon for predictions. |
| `_llmBackbone` | LLM backbone type. |
| `_llmDimension` | LLM hidden dimension. |
| `_lossFunction` | The loss function used for training. |
| `_numFeatures` | Number of input features. |
| `_numHeads` | Number of attention heads. |
| `_numLayers` | Number of reprogramming layers. |
| `_numPatches` | Number of patches. |
| `_numPrototypes` | Number of text prototypes. |
| `_optimizer` | The optimizer used for training. |
| `_outputProjection` | Reference to the output projection layer. |
| `_patchEmbedding` | Reference to the patch embedding layer. |
| `_patchLength` | Patch length for input segmentation. |
| `_patchStride` | Patch stride. |
| `_reprogrammingLayers` | References to the reprogramming attention layers. |
| `_revinMean` | RevIN (reversible instance normalization, Kim et al. |
| `_useNativeMode` | Indicates whether the model is running in native mode (true) or ONNX mode (false). |

