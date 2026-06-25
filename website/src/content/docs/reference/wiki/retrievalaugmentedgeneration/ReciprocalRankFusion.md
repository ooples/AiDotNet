---
title: "ReciprocalRankFusion<T>"
description: "Reciprocal Rank Fusion for combining multiple ranking lists."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.RerankingStrategies`

Reciprocal Rank Fusion for combining multiple ranking lists.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ReciprocalRankFusion(Int32)` | Initializes a new instance of the `ReciprocalRankFusion` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModifiesScores` | Gets a value indicating whether this reranker modifies relevance scores. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FuseRankings(List<List<Document<>>>,Int32)` | Fuses multiple ranking lists using reciprocal rank fusion. |
| `RerankCore(String,IList<Document<>>)` | Reranks documents using reciprocal rank fusion. |

