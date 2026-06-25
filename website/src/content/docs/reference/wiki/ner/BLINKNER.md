---
title: "BLINKNER<T>"
description: "BLINK: BERT-based bi-encoder for entity linking and Named Entity Recognition."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NER.TransformerBased`

BLINK: BERT-based bi-encoder for entity linking and Named Entity Recognition.

## For Beginners

BLINK goes beyond just finding entity mentions - it links them to
specific entries in a knowledge base (like Wikipedia). For example, given "Obama visited
Paris", it not only finds "Obama" (person) and "Paris" (location) but links "Obama" to
the Wikipedia article about Barack Obama and "Paris" to the article about Paris, France
(not Paris, Texas). This disambiguation is done by comparing embeddings of the mention
context with embeddings of entity descriptions.

## How It Works

BLINK (Wu et al., EMNLP 2020 - "Scalable Zero-shot Entity Linking with Dense Entity Retrieval")
is a bi-encoder architecture for entity linking that uses BERT to encode both entity mentions
and entity descriptions, then links mentions to knowledge base entries via dense retrieval.

**Key Innovation - Bi-Encoder Architecture:**
BLINK uses two independent BERT encoders:

1. **Mention Encoder:** Encodes the entity mention with its surrounding context

Input: "[CLS] left context [Ms] mention [Me] right context [SEP]"
Output: 768-dimensional mention embedding

2. **Entity Encoder:** Encodes each knowledge base entity using its title and description

Input: "[CLS] entity_title [ENT] entity_description [SEP]"
Output: 768-dimensional entity embedding

**Two-Stage Linking:**

- **Stage 1 - Candidate Retrieval:** Use FAISS (approximate nearest neighbor search) to

find the top-K entity candidates whose embeddings are closest to the mention embedding.
This is extremely fast (milliseconds for millions of entities) because entity embeddings
are pre-computed and indexed.

- **Stage 2 - Cross-Encoder Re-ranking:** A cross-encoder (BERT that sees both mention

and entity description together) re-ranks the top-K candidates for final prediction.

**Zero-Shot Entity Linking:**
BLINK can link to entities never seen during training because it compares mention
representations with entity description representations in a shared embedding space.
If a new entity is added to the knowledge base with a description, BLINK can link to
it without retraining.

**Performance:**

- AIDA-CoNLL (in-domain): ~87.5% accuracy
- Zero-shot (unseen entities): ~82.3% accuracy
- Cross-lingual entity linking: ~78-85% accuracy

**NER + Entity Linking Pipeline:**
In the combined pipeline, a separate NER model (e.g., BERT-NER) first identifies entity
mentions, then BLINK links each mention to its corresponding knowledge base entry.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BLINKNER(NeuralNetworkArchitecture<>,String,TransformerNEROptions)` | Creates a BLINK model in ONNX inference mode. |
| `BLINKNER(NeuralNetworkArchitecture<>,TransformerNEROptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a BLINK model in native training mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |

