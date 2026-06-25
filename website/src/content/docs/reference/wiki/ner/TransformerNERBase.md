---
title: "TransformerNERBase<T>"
description: "Base class for transformer-based NER models (BERT-NER, RoBERTa-NER, DeBERTa-NER, etc.)."
section: "API Reference"
---

`Base Classes` · `AiDotNet.NER.TransformerBased`

Base class for transformer-based NER models (BERT-NER, RoBERTa-NER, DeBERTa-NER, etc.).

## For Beginners

Transformer models are the most accurate NER models available.
They use "self-attention" to understand each word by looking at all other words in the
sentence simultaneously. Different transformer variants (BERT, RoBERTa, etc.) are like
different editions of the same textbook - they have the same structure but were trained
slightly differently, leading to different strengths.

## How It Works

All transformer-based NER models share the same fine-tuning architecture:

The transformer encoder consists of N stacked layers, each containing:

1. Multi-head self-attention: Each token attends to all other tokens
2. Feed-forward network: Non-linear transformation of each token independently
3. Layer normalization and residual connections: Stabilize training

The key differences between BERT, RoBERTa, DeBERTa, etc. are in the pre-training
(masked language model, replaced token detection, etc.) and attention mechanism
(absolute vs relative position encoding, disentangled attention, etc.). The NER
fine-tuning architecture is identical across all variants.

This base class provides the common implementation. Derived classes only need to:

1. Pass their model-specific name to the base constructor
2. Optionally override CreateDefaultLayers for architecture-specific layer configurations

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TransformerNERBase(NeuralNetworkArchitecture<>,String,TransformerNEROptions,String,String)` | Creates a transformer NER model in ONNX inference mode. |
| `TransformerNERBase(NeuralNetworkArchitecture<>,TransformerNEROptions,String,String,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a transformer NER model in native training mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ExpectedInputShape` |  |
| `NEROptions` | Gets the options for this transformer NER model. |
| `UseNativeMode` | Gets whether this model is in native training mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AiDotNet#NER#Interfaces#INERModel{T}#GetModelSummary` |  |
| `AiDotNet#NER#Interfaces#INERModel{T}#PredictBatch(IEnumerable<Tensor<>>)` |  |
| `AiDotNet#NER#Interfaces#INERModel{T}#TrainAsync(Tensor<>,Tensor<>,Int32,IProgress<NERTrainingProgress>,CancellationToken)` |  |
| `AiDotNet#NER#Interfaces#INERModel{T}#ValidateInputShape(Tensor<>)` |  |
| `ComputeEmissionScores(Tensor<>)` |  |
| `CreateDefaultLayers` | Creates the default layer stack for this transformer NER model. |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Dispose(Boolean)` |  |
| `GetModelMetadata` |  |
| `InitializeLayers` |  |
| `PostprocessOutput(Tensor<>)` |  |
| `PredictLabels(Tensor<>)` |  |
| `PreprocessTokens(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

