---
title: "IFinancialNLPModel<T>"
description: "IFinancialNLPModel<T> — Interfaces in AiDotNet.Finance.Interfaces."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Finance.Interfaces`

_No summary documentation available yet._

## Properties

| Property | Summary |
|:-----|:--------|
| `HiddenDimension` | Gets the hidden dimension of the model. |
| `MaxSequenceLength` | Gets the maximum sequence length (in tokens) that the model can process. |
| `NumSentimentClasses` | Gets the number of sentiment classes the model predicts. |
| `SupportsTraining` | Gets whether training is supported (only in native mode). |
| `UseNativeMode` | Gets whether this model uses native mode (true) or ONNX mode (false). |
| `VocabularySize` | Gets the vocabulary size of the model's tokenizer. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AnalyzeSentiment(String[])` | Analyzes sentiment from raw text strings. |
| `AnalyzeSentiment(Tensor<>)` | Analyzes sentiment from tokenized input. |
| `Detokenize(Int32[])` | Converts token IDs back to text. |
| `GetEmbeddings(Tensor<>)` | Gets embeddings (vector representations) for input tokens. |
| `GetFinancialMetrics` | Gets financial-specific NLP metrics from the model. |
| `GetSequenceEmbedding(Tensor<>)` | Gets the [CLS] token embedding representing the entire input sequence. |
| `Tokenize(String,Nullable<Int32>)` | Tokenizes raw text into token IDs. |

