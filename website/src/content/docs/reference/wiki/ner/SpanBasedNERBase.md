---
title: "SpanBasedNERBase<T>"
description: "Base class for span-based NER models (SpERT, BiaffineNER, PURE)."
section: "API Reference"
---

`Base Classes` · `AiDotNet.NER.SpanBased`

Base class for span-based NER models (SpERT, BiaffineNER, PURE).

## How It Works

Span-based NER models enumerate candidate entity spans (contiguous subsequences) and
classify each span as an entity type or non-entity. This approach differs fundamentally
from sequence labeling (BIO tagging):

**Sequence Labeling (BiLSTM-CRF):**
Labels each token independently: [B-PER, I-PER, O, O, B-ORG, I-ORG]
Cannot naturally handle nested entities (e.g., "New York" inside "New York University")

**Span-Based (SpERT, BiaffineNER, PURE):**
Enumerates spans: (0,1)="Barack", (0,2)="Barack Obama", (2,3)="was", ...
Classifies each span: (0,2)=PER, (4,6)=LOC, (0,6)=non-entity, ...
Naturally handles nested entities because different spans can have different labels

**Architecture:**

The span representation combines boundary tokens, span content, and span width features
into a fixed-size vector for classification.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SpanBasedNERBase(NeuralNetworkArchitecture<>,SpanBasedNEROptions,String,String,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a span-based NER model in native training mode. |
| `SpanBasedNERBase(NeuralNetworkArchitecture<>,String,SpanBasedNEROptions,String,String)` | Creates a span-based NER model in ONNX inference mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ExpectedInputShape` |  |
| `NEROptions` | Gets the options for this span-based NER model. |
| `UseNativeMode` | Gets whether this model is in native training mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AiDotNet#NER#Interfaces#INERModel{T}#GetModelSummary` |  |
| `AiDotNet#NER#Interfaces#INERModel{T}#PredictBatch(IEnumerable<Tensor<>>)` |  |
| `AiDotNet#NER#Interfaces#INERModel{T}#TrainAsync(Tensor<>,Tensor<>,Int32,IProgress<NERTrainingProgress>,CancellationToken)` |  |
| `AiDotNet#NER#Interfaces#INERModel{T}#ValidateInputShape(Tensor<>)` |  |
| `ComputeEmissionScores(Tensor<>)` |  |
| `CreateDefaultLayers` | Creates the default layer stack for this span-based NER model. |
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

