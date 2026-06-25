---
title: "CrossEncoderReranker<T>"
description: "Reranks documents using a cross-encoder model that computes fine-grained relevance scores."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.Rerankers`

Reranks documents using a cross-encoder model that computes fine-grained relevance scores.

## For Beginners

Cross-encoder rerankers make search results much better.

Think of the difference between these two approaches:

**Regular Similarity (Bi-encoder):**

- Converts query to numbers: "best pizza" → [0.2, 0.8, 0.1, ...]
- Converts each document to numbers separately
- Compares numbers to find similar ones
- Fast but misses nuances

**Cross-Encoder Reranking:**

- Takes query + document together: "best pizza" + "Mario's serves authentic Italian pizza"
- Model reads both at once, understanding how they relate
- Produces a precise relevance score
- Slower but much more accurate

Real-world workflow:

1. Initial retrieval: Fast method gets 100 candidates (e.g., vector search or BM25)
2. Reranking: Cross-encoder carefully scores top 10-20 candidates
3. Final results: Reordered by precise relevance scores

Why this works so well:

- Initial retrieval casts a wide net (high recall)
- Reranking refines the results (high precision)
- Best of both worlds: Speed + Accuracy

Common use cases:

- E-commerce search: Find the most relevant products
- Question answering: Find the paragraph that actually answers the question
- Document search: Rank by true relevance, not just keyword overlap

Performance impact:

- Reranking 10-20 docs: Fast enough for real-time
- Reranking 1000s of docs: Too slow, only rerank top candidates

## How It Works

Cross-encoder reranking is the gold standard for improving retrieval quality. Unlike bi-encoders
that encode query and document separately, cross-encoders process the query-document pair together,
allowing for richer interaction and more accurate relevance scoring. This is typically done as a
second-stage reranking step after initial retrieval.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CrossEncoderReranker(Func<String,String,>,Int32)` | Initializes a new instance of the CrossEncoderReranker class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModifiesScores` | Gets a value indicating whether this reranker modifies relevance scores. |

## Methods

| Method | Summary |
|:-----|:--------|
| `RerankCore(String,IList<Document<>>)` | Reranks documents using the cross-encoder model. |

