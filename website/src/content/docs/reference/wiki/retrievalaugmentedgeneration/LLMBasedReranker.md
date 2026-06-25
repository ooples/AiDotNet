---
title: "LLMBasedReranker<T>"
description: "LLM-based reranking using language model relevance assessment."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.RerankingStrategies`

LLM-based reranking using language model relevance assessment.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LLMBasedReranker(String,String)` | Initializes a new instance of the `LLMBasedReranker` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModifiesScores` | Gets a value indicating whether this reranker modifies relevance scores. |

## Methods

| Method | Summary |
|:-----|:--------|
| `RerankCore(String,IList<Document<>>)` | Reranks documents using LLM-based relevance scoring. |

