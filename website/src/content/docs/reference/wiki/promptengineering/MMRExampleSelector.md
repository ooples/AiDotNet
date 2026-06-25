---
title: "MMRExampleSelector<T>"
description: "Selects examples using Maximum Marginal Relevance (MMR) to balance relevance and diversity."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PromptEngineering.FewShot`

Selects examples using Maximum Marginal Relevance (MMR) to balance relevance and diversity.

## For Beginners

Picks examples that are both relevant AND different from each other.

Think of it like building a playlist:

- Pure relevance: All songs sound almost identical (boring!)
- Pure diversity: Random songs that don't fit together (confusing!)
- MMR: Similar style, but each song brings something different (perfect!)

Example:

The lambda parameter controls the balance:

- lambda = 1.0: Pure relevance (like SemanticSimilarity)
- lambda = 0.5: Equal balance
- lambda = 0.0: Pure diversity (like Diversity)

## How It Works

MMR combines relevance (similarity to query) and diversity (difference from already-selected examples).
It iteratively selects examples that maximize: lambda * relevance - (1 - lambda) * max_similarity_to_selected

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MMRExampleSelector(Func<String,Vector<>>)` | Initializes a new instance of the MMRExampleSelector class. |
| `MMRExampleSelector(Func<String,Vector<>>,)` | Initializes a new instance of the MMRExampleSelector class. |
| `MMRExampleSelector(IEmbeddingModel<>)` | Initializes a new instance of the MMRExampleSelector class. |
| `MMRExampleSelector(IEmbeddingModel<>,)` | Initializes a new instance of the MMRExampleSelector class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Lambda` | Gets the lambda value used for MMR scoring. |

## Methods

| Method | Summary |
|:-----|:--------|
| `OnExampleAdded(FewShotExample)` | Called when an example is added. |
| `OnExampleRemoved(FewShotExample)` | Called when an example is removed. |
| `SelectExamplesCore(String,Int32)` | Selects examples using the MMR algorithm. |

