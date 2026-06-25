---
title: "Chronos<T>"
description: "Chronos foundation model for time series forecasting using tokenization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Forecasting.Foundation`

Chronos foundation model for time series forecasting using tokenization.

## For Beginners

Chronos brings the power of language models to time series:

**The Key Insight:**
Language models like GPT are amazing at predicting the next word in a sequence.
Chronos asks: "What if we convert time series to 'words' and use the same approach?"

**How Tokenization Works:**

1. **Scaling**: Normalize values (e.g., mean=0, std=1 or min-max to [-1, 1])
2. **Quantization**: Divide range into bins (e.g., 4096 bins)
3. **Token Assignment**: Each value gets the bin number as its "token"

Example: If value = 0.73 and bins are [-1 to 1] with 100 bins:

- 0.73 falls in bin 86 (because 0.73 is 86% of the way from -1 to 1)
- Token = 86

**Why This Works:**

- LLMs excel at pattern recognition in sequences
- Time series have patterns just like language
- Pretrained LLMs already understand "sequences" - just need different tokens
- Can leverage massive pretraining investments

**Sampling for Uncertainty:**
Like GPT sampling text, Chronos samples from predicted token probabilities:

- Take 20 samples → 20 different forecasts
- Median = point forecast
- Spread = uncertainty estimate

**Model Sizes:**

- Mini (20M params): Fast, good for experiments
- Small (46M params): Balanced
- Base (200M params): Strong general use
- Large (710M params): Best accuracy

## How It Works

Chronos is Amazon's foundation model that treats time series forecasting as a language
modeling problem. It tokenizes continuous time series values into discrete tokens,
uses a pretrained language model (T5-style) to predict future tokens, and then
converts tokens back to continuous values.

**Reference:** Ansari et al., "Chronos: Learning the Language of Time Series", 2024.
https://arxiv.org/abs/2403.07815

**Thread Safety:** This class is NOT thread-safe. Concurrent calls to
`Double[])` or `Tensor{`
will result in undefined behavior due to shared tokenization state.
Create separate instances for concurrent usage scenarios.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Chronos(NeuralNetworkArchitecture<>,ChronosFinanceOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a Chronos model in native mode for training or fine-tuning. |
| `Chronos(NeuralNetworkArchitecture<>,String,ChronosFinanceOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a Chronos model using pretrained ONNX model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsChannelIndependent` |  |
| `MaxContextLength` |  |
| `MaxPredictionHorizon` |  |
| `ModelSize` |  |
| `NumFeatures` |  |
| `NumTokens` | Gets the number of discrete tokens used for quantization. |
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
| `CaptureTokenScale(Tensor<>)` | Captures the per-input min/range used by `Tensor{` to rebuild continuous values. |
| `ConcatenatePredictions(List<Tensor<>>,Int32)` | Concatenates multiple prediction tensors for extended horizons. |
| `CreateNewInstance` | Creates a new instance of this model with the same configuration. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Reads Chronos-specific configuration during deserialization. |
| `Detokenize(Tensor<>)` | Detokenizes model output (logits) back to continuous values. |
| `Dispose(Boolean)` | Disposes resources used by the Chronos model. |
| `Evaluate(Tensor<>,Tensor<>)` |  |
| `ExtractLayerReferences` | Extracts references to specific layers from the layer collection. |
| `Forecast(Tensor<>,Double[])` |  |
| `ForecastNative(Tensor<>)` | Performs native mode forecasting. |
| `ForecastOnnx(Tensor<>)` | Performs ONNX mode forecasting. |
| `Forward(Tensor<>)` | Performs the forward pass through Chronos. |
| `GenerateQuantileSamples(Tensor<>,Double[])` | Generates quantile samples by sampling from token distributions. |
| `GetFinancialMetrics` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the neural network layers for Chronos. |
| `PredictCore(Tensor<>)` |  |
| `SampleFromLogits(Tensor<>,Int32,Random)` | Samples a token index from logits using temperature-scaled softmax. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Writes Chronos-specific configuration during serialization. |
| `ShiftInputWithPredictions(Tensor<>,Tensor<>,Int32)` | Shifts input tensor by incorporating predictions for autoregressive forecasting. |
| `Tokenize(Tensor<>)` | Tokenizes continuous time series values into discrete tokens. |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |
| `ValidateCustomLayers(List<ILayer<>>)` | Validates that custom layers meet Chronos architectural requirements. |
| `ValidateOptions(ChronosFinanceOptions<>)` | Validates the Chronos options. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_finalNorm` | Final layer normalization before the language model head. |
| `_lmHead` | Language model head that predicts token probabilities. |
| `_tokenEmbedding` | Token embedding layer that maps token IDs to vectors. |
| `_transformerLayers` | Transformer layers for processing the embedded sequence. |
| `_useNativeMode` | Indicates whether this network uses native layers (true) or ONNX model (false). |

