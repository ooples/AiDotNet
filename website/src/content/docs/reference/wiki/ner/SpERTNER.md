---
title: "SpERTNER<T>"
description: "SpERT: Span-based Entity and Relation Transformer for joint entity and relation extraction."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NER.SpanBased`

SpERT: Span-based Entity and Relation Transformer for joint entity and relation extraction.

## For Beginners

SpERT looks at all possible groups of consecutive words (spans) in a
sentence and classifies each as an entity type or non-entity. Unlike BiLSTM-CRF which labels
each word one at a time, SpERT considers entire phrases at once. It can also find relationships
between entities (e.g., "born in" between a person and a location).

## How It Works

SpERT (Eberts and Ulges, ECAI 2020 - "Span-based Joint Entity and Relation Extraction with
Transformer Pre-training") performs joint entity recognition and relation extraction using
a span-based approach built on top of a pre-trained transformer encoder.

**Architecture Overview:**

1. **Transformer Encoder:** Pre-trained BERT encodes the input sentence into contextual

token representations

2. **Span Representation:** For each candidate span (i, j), the representation is:

span(i,j) = [h_i; h_j; maxpool(h_i:h_j); width_embedding]
where h_i, h_j are boundary tokens, maxpool is over the span content, and
width_embedding encodes the span length

3. **Entity Classifier:** A feedforward network classifies each span representation

into entity types or non-entity

4. **Relation Classifier:** For each pair of predicted entity spans, a relation

classifier predicts the relation type using concatenated span representations
plus context between the entities

**Negative Sampling:**
Since most spans are non-entities and most entity pairs have no relation, SpERT uses
careful negative sampling during training. The ratio of negative to positive samples
is a key hyperparameter (typically 100:1 for entities).

**Performance:**

- CoNLL-2004: 86.3% entity F1, 72.9% relation F1
- ADE dataset: 89.3% entity F1, 79.2% relation F1
- SciERC: 70.3% entity F1, 48.4% relation F1

**Key Insight:**
By operating on spans rather than individual tokens, SpERT avoids BIO label constraint
issues and naturally handles multi-token entities. The joint entity-relation extraction
ensures that entity and relation decisions are mutually informed.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SpERTNER(NeuralNetworkArchitecture<>,SpanBasedNEROptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a SpERT model in native training mode. |
| `SpERTNER(NeuralNetworkArchitecture<>,String,SpanBasedNEROptions)` | Creates a SpERT model in ONNX inference mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateDefaultLayers` |  |
| `CreateNewInstance` |  |

