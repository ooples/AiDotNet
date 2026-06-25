---
title: "IReranker<T>"
description: "Defines the contract for reranking retrieved documents to improve relevance ordering."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for reranking retrieved documents to improve relevance ordering.

## For Beginners

A reranker is like a second opinion on search results.

Think of it like a two-stage hiring process:

Stage 1 (Initial Retrieval):

- Quick screening of 1000 applicants
- Filter to top 20 based on resume keywords
- Fast but might miss some good candidates

Stage 2 (Reranking):

- Detailed review of those 20 candidates
- Deeper analysis of experience and fit
- Slower but more accurate
- Final ranking of best 5

Similarly, reranking takes the initial search results and re-orders them using
more sophisticated analysis, ensuring the best results appear first.

## How It Works

A reranker refines the ordering of initially retrieved documents using more sophisticated
relevance scoring. While initial retrieval must be fast and may use simple similarity metrics,
reranking can employ computationally expensive methods like cross-encoders or large language
models to achieve better relevance rankings for the final result set.

## Properties

| Property | Summary |
|:-----|:--------|
| `ModifiesScores` | Gets a value indicating whether this reranker modifies relevance scores. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Rerank(String,IEnumerable<Document<>>)` | Reranks a collection of documents based on their relevance to a query. |
| `Rerank(String,IEnumerable<Document<>>,Int32)` | Reranks documents and returns only the top-k highest scoring results. |

