---
title: "RerankerBase<T>"
description: "Provides a base implementation for document rerankers with common functionality."
section: "API Reference"
---

`Base Classes` · `AiDotNet.RetrievalAugmentedGeneration.Rerankers`

Provides a base implementation for document rerankers with common functionality.

## For Beginners

This is the foundation that all reranking methods build upon.

Think of it like a template for improving search results:

- It handles common tasks (checking inputs, limiting results, normalizing scores)
- Specific reranking methods (cross-encoder, LLM-based) just fill in how they score documents
- This ensures all rerankers work consistently

## How It Works

This abstract class implements the IReranker interface and provides common functionality
for reranking strategies. It handles validation, result limiting, and score normalization
while allowing derived classes to focus on implementing the core reranking algorithm.

## Properties

| Property | Summary |
|:-----|:--------|
| `ModifiesScores` | Gets a value indicating whether this reranker modifies relevance scores. |

## Methods

| Method | Summary |
|:-----|:--------|
| `NormalizeScores(IList<Document<>>)` | Normalizes relevance scores to the 0-1 range. |
| `Rerank(String,IEnumerable<Document<>>)` | Reranks a collection of documents based on their relevance to a query. |
| `Rerank(String,IEnumerable<Document<>>,Int32)` | Reranks documents and returns only the top-k highest scoring results. |
| `RerankCore(String,IList<Document<>>)` | Core reranking logic to be implemented by derived classes. |
| `ValidateDocuments(IEnumerable<Document<>>)` | Validates the document collection. |
| `ValidateQuery(String)` | Validates the query string. |
| `ValidateTopK(Int32)` | Validates the topK parameter. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Provides mathematical operations for the numeric type T. |

