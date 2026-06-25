---
title: "XLMRoBERTaNER<T>"
description: "XLM-RoBERTa-NER: Cross-lingual RoBERTa for multilingual Named Entity Recognition."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NER.TransformerBased`

XLM-RoBERTa-NER: Cross-lingual RoBERTa for multilingual Named Entity Recognition.

## For Beginners

XLM-RoBERTa can do NER in over 100 languages. The remarkable thing
is you can train it on English data and it will recognize entities in other languages too.
Use this when you need NER for non-English text or multilingual applications.

## How It Works

XLM-RoBERTa-NER (Conneau et al., ACL 2020 - "Unsupervised Cross-lingual Representation
Learning at Scale") fine-tunes a multilingual transformer for NER across 100+ languages.

**Key Features:**

- **100+ languages:** Pre-trained on 2.5TB of CommonCrawl data in 100 languages
- **Cross-lingual transfer:** Fine-tune on English NER data, apply to any supported language
- **SentencePiece tokenization:** Shared vocabulary across all languages (250K tokens)
- **RoBERTa architecture:** Same architecture as RoBERTa but with multilingual pre-training

**Zero-shot Cross-lingual NER:**
XLM-RoBERTa enables zero-shot cross-lingual NER: train on English CoNLL-2003, then
predict entities in German, Spanish, Dutch, etc. without any target-language training data.
This is possible because the multilingual pre-training creates a shared representation space
where similar concepts in different languages have similar embeddings.

**Performance:**

- English CoNLL-2003: ~92.5% F1
- German (zero-shot from English): ~79% F1
- Spanish (zero-shot from English): ~81% F1
- Cross-lingual average: ~85% F1 (significantly better than mBERT)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `XLMRoBERTaNER(NeuralNetworkArchitecture<>,String,TransformerNEROptions)` | Creates an XLM-RoBERTa-NER model in ONNX inference mode. |
| `XLMRoBERTaNER(NeuralNetworkArchitecture<>,TransformerNEROptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates an XLM-RoBERTa-NER model in native training mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |

