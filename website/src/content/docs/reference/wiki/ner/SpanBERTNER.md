---
title: "SpanBERTNER<T>"
description: "SpanBERT-NER: Span-level BERT pre-training with token classification for NER."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NER.TransformerBased`

SpanBERT-NER: Span-level BERT pre-training with token classification for NER.

## For Beginners

SpanBERT is a BERT variant designed for tasks involving spans of text,
like NER. While regular BERT masks individual words during training, SpanBERT masks groups
of words together. This teaches it to understand multi-word entities like "New York City" or
"Goldman Sachs" as single units, which directly helps with NER accuracy.

## How It Works

SpanBERT-NER (Joshi et al., TACL 2020 - "SpanBERT: Improving Pre-training by Representing
and Predicting Spans") uses a BERT variant specifically designed for span-level tasks like NER.

**Key Innovations:**

- **Span masking:** Instead of masking random individual tokens, SpanBERT masks contiguous

spans of tokens (geometric distribution, mean length 3.8). This forces the model to learn
better span-level representations, directly beneficial for NER where entities are spans.

- **Span boundary objective (SBO):** The model predicts masked tokens using the span

boundary tokens (positions just before and after the span), encouraging the model to encode
span information at boundary positions. This is particularly useful for detecting entity
boundaries (B- tags) in NER.

- **No NSP:** Removes next sentence prediction, following RoBERTa's findings.

**Why SpanBERT excels at NER:**
NER is fundamentally a span-level task: entities are contiguous spans of tokens. SpanBERT's
span masking pre-training teaches the model to understand multi-token entities as units,
rather than treating each token independently. The span boundary objective ensures that
B-tag positions (entity starts) contain strong span representations.

**Performance (CoNLL-2003):**

- SpanBERT-base: ~92.8% F1
- SpanBERT-large: ~93.4% F1

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SpanBERTNER(NeuralNetworkArchitecture<>,String,TransformerNEROptions)` | Creates a SpanBERT-NER model in ONNX inference mode. |
| `SpanBERTNER(NeuralNetworkArchitecture<>,TransformerNEROptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a SpanBERT-NER model in native training mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |

