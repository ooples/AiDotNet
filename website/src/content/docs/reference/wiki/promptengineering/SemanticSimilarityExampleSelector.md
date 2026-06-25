---
title: "SemanticSimilarityExampleSelector<T>"
description: "Selects examples based on semantic similarity to the query."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PromptEngineering.FewShot`

Selects examples based on semantic similarity to the query.

## For Beginners

Picks examples most similar in meaning to your query.

How it works:

1. Convert query and examples to mathematical vectors (embeddings)
2. Calculate similarity between query and each example
3. Return the most similar examples

Example:

Use this when:

- Query types vary significantly
- Relevant examples improve performance
- You have an embedding model available

## How It Works

This selector uses embedding vectors to find examples that are semantically similar to the query.
It converts both the query and examples into vector representations and selects those with the
highest cosine similarity.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SemanticSimilarityExampleSelector(Func<String,Vector<>>)` | Initializes a new instance of the SemanticSimilarityExampleSelector class. |
| `SemanticSimilarityExampleSelector(IEmbeddingModel<>)` | Initializes a new instance of the SemanticSimilarityExampleSelector class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `OnExampleAdded(FewShotExample)` | Called when an example is added. |
| `OnExampleRemoved(FewShotExample)` | Called when an example is removed. |
| `SelectExamplesCore(String,Int32)` | Selects the most semantically similar examples. |

