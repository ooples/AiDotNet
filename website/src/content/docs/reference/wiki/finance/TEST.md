---
title: "TEST<T>"
description: "TEST — Text Prototype Aligned Embedding to Activate LLM's Ability for Time Series."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Forecasting.Foundation`

TEST — Text Prototype Aligned Embedding to Activate LLM's Ability for Time Series.

## For Beginners

TEST bridges the gap between language models and time series
by creating text descriptions of temporal patterns (like "rising trend with weekly
seasonality") and teaching the model to align numerical patterns with these descriptions.
This lets a language model understand time series without converting numbers to text,
combining the best of both worlds.

## How It Works

TEST generates text-prototype-aligned embeddings for time series by leveraging pretrained
language model knowledge. It translates seasonal/trend patterns into text descriptions
and aligns time series embeddings to these text prototypes via contrastive learning.

**Reference:** Sun et al., "TEST: Text Prototype Aligned Embedding to Activate LLM's Ability for Time Series", 2024.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TEST(NeuralNetworkArchitecture<>,String,TESTOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a TEST model using a pretrained ONNX model. |
| `TEST(NeuralNetworkArchitecture<>,TESTOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a TEST model in native mode for training or fine-tuning. |

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
| `GetFinancialMetrics` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `PredictCore(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

