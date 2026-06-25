---
title: "TinyBERTNER<T>"
description: "TinyBERT-NER: Two-stage distilled BERT for ultra-efficient Named Entity Recognition."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NER.TransformerBased`

TinyBERT-NER: Two-stage distilled BERT for ultra-efficient Named Entity Recognition.

## For Beginners

TinyBERT is one of the smallest BERT models available. It uses a
sophisticated two-step learning process to compress BERT down to a fraction of its size
while retaining useful accuracy. Use TinyBERT-NER for edge deployment, mobile applications,
or when you need the fastest possible NER with reasonable accuracy.

## How It Works

TinyBERT-NER (Jiao et al., EMNLP 2020 Findings - "TinyBERT: Distilling BERT for Natural
Language Understanding") uses a novel two-stage knowledge distillation framework to create
an extremely compact BERT model (~7.5x smaller and ~9.4x faster than BERT-base).

**Two-Stage Distillation:**

- **Stage 1 - General Distillation:** Distill BERT's general language knowledge using

a large unlabeled corpus. Transfers attention matrices, hidden states, and embedding
layer knowledge from teacher to student.

- **Stage 2 - Task-Specific Distillation:** Fine-tune the student on task-specific data

(e.g., NER) using both the labeled data and the teacher's task-specific knowledge.

**Distillation Losses (Layer-by-Layer):**

- **Embedding loss:** MSE between teacher and student embedding layers
- **Attention loss:** MSE between teacher and student attention matrices
- **Hidden state loss:** MSE between teacher and student hidden representations
- **Prediction loss:** Cross-entropy between teacher and student output distributions

**TinyBERT-4L Architecture:**

- 4 transformer layers (vs 12 in BERT-base)
- 312 hidden dimension (vs 768 in BERT-base)
- 12 attention heads (maintained for knowledge transfer)
- ~14.5M parameters (vs 110M in BERT-base)

**Performance:**

- NER (CoNLL-2003): ~89.5% F1 (vs BERT-base ~92.4%)
- 7.5x smaller than BERT-base
- 9.4x faster inference

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TinyBERTNER(NeuralNetworkArchitecture<>,String,TransformerNEROptions)` | Creates a TinyBERT-NER model in ONNX inference mode. |
| `TinyBERTNER(NeuralNetworkArchitecture<>,TransformerNEROptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a TinyBERT-NER model in native training mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |

