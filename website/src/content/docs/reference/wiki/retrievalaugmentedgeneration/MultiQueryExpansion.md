---
title: "MultiQueryExpansion"
description: "Expands queries by generating multiple query variations from different perspectives using LLM-based reformulation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.QueryExpansion`

Expands queries by generating multiple query variations from different perspectives using LLM-based reformulation.

## For Beginners

Think of this like asking the same question in different ways:

Regular search: "machine learning algorithms"

Multi-query expansion generates:

1. "What are machine learning algorithms?" (question form)
2. "information about machine learning algorithms" (contextual)
3. "artificial intelligence techniques" (synonym expansion)
4. "ML algorithms" (simplified)
5. "machine learning methods" (variation)

Then searches using ALL variations and combines results!

```cs
var expander = new MultiQueryExpansion(
llmEndpoint: "http://localhost:1234/v1",
llmApiKey: "your-key",
numVariations: 5
);

var queries = expander.ExpandQuery("deep learning optimization");
// Returns: ["deep learning optimization", "What is deep learning optimization?", 
// "details about deep learning optimization", "neural network training", ...]
```

Why use MultiQueryExpansion:

- Finds documents using different terminology (e.g., "car" vs "automobile")
- Improves recall without sacrificing precision
- Handles ambiguous queries (explores multiple interpretations)
- Effective for cross-domain search (technical ↔ layman terms)

When NOT to use it:

- Very specific queries with clear terminology (wastes compute)
- High-latency systems (multiplies retrieval cost by numVariations)
- When you need ONLY exact matches
- Extremely short queries (no room for variation)

## How It Works

MultiQueryExpansion addresses the vocabulary mismatch problem in retrieval by creating diverse phrasings of the same
information need. Instead of searching with a single query, it generates 3-5 reformulations (questions, statements,
contextual phrases, synonyms) and retrieves documents for each variation, then merges and deduplicates results.
This approach significantly improves recall by capturing documents that use different terminology than the original
query. The implementation uses pattern-based transformations as a fallback but is designed to integrate with LLMs
for more sophisticated reformulations (e.g., technical → layman, abstract → concrete, general → specific perspectives).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MultiQueryExpansion(String,String,Int32)` | Initializes a new instance of the `MultiQueryExpansion` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ExpandQuery(String)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `RegexTimeout` | Timeout for regex operations to prevent ReDoS attacks. |

