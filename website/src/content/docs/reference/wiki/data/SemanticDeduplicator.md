---
title: "SemanticDeduplicator"
description: "Detects semantic duplicates using embedding cosine similarity."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Quality`

Detects semantic duplicates using embedding cosine similarity.

## How It Works

Semantic deduplication finds documents with the same meaning even if worded differently.
Requires pre-computed embeddings (e.g., from a sentence transformer).
More expensive than MinHash but catches paraphrased duplicates.

## Methods

| Method | Summary |
|:-----|:--------|
| `CosineSimilarity(Double[],Double[])` | Computes cosine similarity between two embedding vectors. |
| `FindDuplicates(Double[][])` | Finds duplicate indices from pre-computed embeddings. |

