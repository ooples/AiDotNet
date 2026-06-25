---
title: "PURENER<T>"
description: "PURE: Princeton University Relation Extraction - pipeline approach for joint entity and relation extraction."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NER.SpanBased`

PURE: Princeton University Relation Extraction - pipeline approach for joint entity and relation extraction.

## For Beginners

PURE takes a refreshingly simple approach to entity extraction.
Instead of building a complex model that tries to do everything at once, it uses a two-step
pipeline: first find all entities, then find relationships between them. Despite its
simplicity, it achieves state-of-the-art results by leveraging the power of pre-trained
transformers. Use PURE when you want a simple, effective span-based NER approach.

## How It Works

PURE (Zhong and Chen, NAACL 2021 - "A Frustratingly Easy Approach for Entity and Relation
Extraction") demonstrates that a simple pipeline approach to entity and relation extraction
can match or outperform complex joint models. The key insight is that modern pre-trained
transformers are so powerful that complex architectural innovations provide diminishing returns.

**Pipeline Architecture (Two Independent Models):****Stage 1 - Entity Model:**

- Input: sentence with token embeddings from pre-trained transformer
- For each candidate span (i, j), compute span representation:

span(i,j) = [h_i; h_j; span_width_embedding]

- Classify each span as an entity type or non-entity
- Output: set of typed entity spans

**Stage 2 - Relation Model (optional, not used for NER-only):**

- Input: sentence with entity span markers from Stage 1
- For each pair of predicted entities, classify the relation type
- Uses entity markers ([E1], [/E1], [E2], [/E2]) inserted around entities
- Output: set of typed relations between entity pairs

**Why "Frustratingly Easy":**
Previous work assumed that joint models (predicting entities and relations simultaneously)
were necessary for strong performance because entity and relation decisions are interdependent.
PURE shows that with modern pre-trained transformers, a simple pipeline that first predicts
entities and then predicts relations achieves comparable or better results.

**Cross-Sentence Context:**
PURE optionally uses cross-sentence context by extending the input window beyond the
current sentence, which helps for entities whose types depend on broader context.

**Performance:**

- ACE 2005 Entity: ~89.7% F1 (state-of-the-art)
- ACE 2005 Relation: ~69.0% F1
- SciERC Entity: ~68.9% F1
- SciERC Relation: ~38.5% F1

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PURENER(NeuralNetworkArchitecture<>,SpanBasedNEROptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a PURE model in native training mode. |
| `PURENER(NeuralNetworkArchitecture<>,String,SpanBasedNEROptions)` | Creates a PURE model in ONNX inference mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateDefaultLayers` |  |
| `CreateNewInstance` |  |

