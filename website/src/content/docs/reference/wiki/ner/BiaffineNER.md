---
title: "BiaffineNER<T>"
description: "Biaffine-NER: Named Entity Recognition as dependency parsing using biaffine classifiers."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NER.SpanBased`

Biaffine-NER: Named Entity Recognition as dependency parsing using biaffine classifiers.

## For Beginners

Biaffine-NER treats entity recognition like finding matching
brackets: for each possible pair of words (start, end), it computes how likely they are
to be the boundaries of an entity. This is more flexible than labeling each word individually
because it can naturally handle overlapping entities (like "New York" being both a city
and part of "New York University").

## How It Works

Biaffine-NER (Yu et al., ACL 2020 - "Named Entity Recognition as Dependency Parsing")
reformulates NER as identifying start and end boundaries of entity spans using biaffine
attention, an approach borrowed from dependency parsing.

**Key Innovation - Biaffine Attention for NER:**
Instead of BIO sequence labeling, Biaffine-NER constructs a span scoring matrix where
entry (i, j, k) represents the score that tokens i through j form an entity of type k.
The biaffine scoring function is:

score(i, j, k) = h_start_i^T * W_k * h_end_j + b_start_i^T * h_start_i + b_end_j^T * h_end_j + bias_k

where:

- h_start_i = MLP_start(encoder(x_i)) transforms the start token representation
- h_end_j = MLP_end(encoder(x_j)) transforms the end token representation
- W_k is a biaffine weight matrix for entity type k
- The biaffine term captures the interaction between start and end representations

**Architecture:**

1. **Encoder:** BERT/BiLSTM produces contextual token representations
2. **Start/End MLPs:** Separate feedforward networks for start and end boundary representations
3. **Biaffine Classifier:** Scores all (start, end, entity-type) triples simultaneously
4. **Decoding:** Select spans with score above threshold, resolve conflicts via greedy/optimal

**Advantages over BIO Tagging:**

- Naturally handles nested entities (overlapping spans get independent scores)
- No label transition constraints needed (no B-I-O consistency issues)
- Efficient: O(n^2 * k) scoring where n = seq length, k = entity types
- Joint boundary detection: start and end predictions are coupled via biaffine interaction

**Performance:**

- CoNLL-2003: ~93.5% F1 (flat NER)
- ACE 2004: ~87.3% F1 (nested NER)
- ACE 2005: ~86.7% F1 (nested NER)
- GENIA: ~79.2% F1 (nested biomedical NER)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BiaffineNER(NeuralNetworkArchitecture<>,SpanBasedNEROptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a Biaffine-NER model in native training mode. |
| `BiaffineNER(NeuralNetworkArchitecture<>,String,SpanBasedNEROptions)` | Creates a Biaffine-NER model in ONNX inference mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateDefaultLayers` |  |
| `CreateNewInstance` |  |

