---
title: "DistilBERTNER<T>"
description: "DistilBERT-NER: Knowledge-distilled BERT for efficient Named Entity Recognition."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NER.TransformerBased`

DistilBERT-NER: Knowledge-distilled BERT for efficient Named Entity Recognition.

## For Beginners

DistilBERT is a compressed version of BERT that runs 60% faster
while keeping 97% of the accuracy. Think of it as a "student" model that learned from
the "teacher" (BERT). Use DistilBERT-NER when you need fast NER with good (but not
maximum) accuracy, especially for production deployments with latency requirements.

## How It Works

DistilBERT-NER (Sanh et al., NeurIPS 2019 Workshop - "DistilBERT, a distilled version of BERT:
smaller, faster, cheaper and lighter") uses knowledge distillation to create a compact BERT
variant that retains 97% of BERT's language understanding while being 60% faster.

**Knowledge Distillation Process:**

- **Teacher:** Full BERT-base model (12 layers, 110M parameters)
- **Student:** DistilBERT (6 layers, 66M parameters - 40% fewer)
- **Training signal:** Combination of distillation loss (soft targets from teacher),

masked language modeling loss, and cosine embedding loss (align hidden states)

- **Architecture:** Removes token-type embeddings and the pooler layer, keeps every other

transformer layer from BERT

**Performance vs Efficiency:**

- 40% smaller than BERT-base (66M vs 110M parameters)
- 60% faster inference (fewer transformer layers)
- Retains 97% of BERT's performance on GLUE benchmark
- NER (CoNLL-2003): ~91.2% F1 (vs BERT-base ~92.4%)

**Trade-offs:**
DistilBERT sacrifices ~1.2% F1 on NER compared to BERT-base, but gains significant speed
and memory improvements. This makes it ideal for production deployments where latency
and resource constraints matter more than the last percentage of accuracy.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DistilBERTNER(NeuralNetworkArchitecture<>,String,TransformerNEROptions)` | Creates a DistilBERT-NER model in ONNX inference mode. |
| `DistilBERTNER(NeuralNetworkArchitecture<>,TransformerNEROptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a DistilBERT-NER model in native training mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |

