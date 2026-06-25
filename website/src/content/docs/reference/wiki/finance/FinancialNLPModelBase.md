---
title: "FinancialNLPModelBase<T>"
description: "Base class for all financial NLP models, implementing the dual-mode pattern."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Finance.Base`

Base class for all financial NLP models, implementing the dual-mode pattern.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FinancialNLPModelBase(NeuralNetworkArchitecture<>,ILossFunction<>,Double)` | Initializes a new instance with deferred NLP configuration. |
| `FinancialNLPModelBase(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,ILossFunction<>)` | Initializes a new NLP model base for training (native mode). |
| `FinancialNLPModelBase(NeuralNetworkArchitecture<>,String,Int32,Int32,Int32,Int32)` | Initializes a new NLP model base from a pretrained ONNX model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `HiddenDimension` |  |
| `MaxSequenceLength` |  |
| `NumSentimentClasses` |  |
| `VocabularySize` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AnalyzeSentiment(String[])` |  |
| `AnalyzeSentiment(Tensor<>)` |  |
| `Detokenize(Int32[])` |  |
| `GetEmbeddings(Tensor<>)` |  |
| `GetFinancialMetrics` |  |
| `GetSequenceEmbedding(Tensor<>)` |  |
| `Tokenize(String,Nullable<Int32>)` |  |

