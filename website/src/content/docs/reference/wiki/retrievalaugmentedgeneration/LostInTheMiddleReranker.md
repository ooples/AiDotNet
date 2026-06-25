---
title: "LostInTheMiddleReranker<T>"
description: "Addresses the \"lost in the middle\" problem by strategically reordering documents."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.RerankingStrategies`

Addresses the "lost in the middle" problem by strategically reordering documents.

## How It Works

Research shows LLMs often ignore information in the middle of long contexts.
This reranker places most relevant documents at the beginning and end of the context.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LostInTheMiddleReranker` | Initializes a new instance of the `LostInTheMiddleReranker` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModifiesScores` | Gets a value indicating whether this reranker modifies relevance scores. |

## Methods

| Method | Summary |
|:-----|:--------|
| `RerankCore(String,IList<Document<>>)` | Reranks documents to avoid the "lost in the middle" problem. |

