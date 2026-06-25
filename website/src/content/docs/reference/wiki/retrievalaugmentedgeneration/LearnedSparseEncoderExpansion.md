---
title: "LearnedSparseEncoderExpansion"
description: "Expands queries using learned sparse representations (SPLADE-like) with term importance weighting for hybrid retrieval."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.QueryExpansion`

Expands queries using learned sparse representations (SPLADE-like) with term importance weighting for hybrid retrieval.

## For Beginners

Think of this like a smart thesaurus that knows which related words actually matter:

Regular thesaurus expansion: "fast" → "quick rapid speedy swift hasty" (all equally)

Learned sparse expansion: "fast" → "quick(0.8) rapid(0.7) speed(0.6)" (weighted by importance)

The weights tell the search how much each term matters!

Example query: "neural network training"

Expansion with weights:

- Original: neural(1.0) network(1.0) training(1.0)
- Expanded: neural(1.0) network(1.0) training(1.0) + networks(0.7) train(0.6) learning(0.8) optimization(0.7)

```cs
var expander = new LearnedSparseEncoderExpansion(
modelPath: "models/splade.onnx",
maxExpansionTerms: 10,
minTermWeight: 0.5 // Only include terms weighted >= 0.5
);

var queries = expander.ExpandQuery("climate change mitigation");
// Returns: ["climate change mitigation", 
// "climate climate change change mitigation global warming reduction carbon"] 
// (term repetition encodes weights)
```

Why use LearnedSparseEncoderExpansion:

- Best of both worlds: keyword precision + semantic expansion
- Learned weights focus on truly relevant terms (not all synonyms)
- Handles domain-specific terminology better than generic expansion
- Effective for technical and scientific queries

When NOT to use it:

- Model not available (requires trained SPLADE/similar model)
- Simple keyword matching is sufficient
- Storage-constrained systems (expanded representations use more space)
- When pure semantic search works well enough (dense retrieval)

## How It Works

LearnedSparseEncoderExpansion combines the benefits of sparse (keyword-based) and dense (semantic) retrieval by
using a learned model to expand queries with semantically related terms weighted by importance. Unlike traditional
query expansion that adds synonyms uniformly, this approach uses neural networks (e.g., SPLADE) to predict term
relevance scores, generating sparse representations where only important expansion terms are included. The model
learns which terms to add and their weights through training on retrieval tasks. This implementation provides a
heuristic-based fallback using term statistics (length, capitalization, occurrence patterns) and morphological
variations, but is designed to load actual SPLADE or similar models for production use. The weighted expansion
improves both recall (finds related documents) and precision (weights focus on relevant terms).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LearnedSparseEncoderExpansion(String,Int32,Double)` | Initializes a new instance of the `LearnedSparseEncoderExpansion` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ExpandQuery(String)` |  |

