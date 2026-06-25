---
title: "SubQueryExpansion"
description: "Expands complex queries by decomposing them into simpler, focused sub-queries for parallel retrieval."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.QueryExpansion`

Expands complex queries by decomposing them into simpler, focused sub-queries for parallel retrieval.

## For Beginners

Think of this like breaking a big question into smaller, easier ones:

Complex query: "Explain machine learning, deep learning, and reinforcement learning"

Sub-query decomposition:

1. "Explain machine learning"
2. "Explain deep learning"
3. "Explain reinforcement learning"
4. "information about machine learning" (key concept)
5. "information about deep learning" (key concept)

Each sub-query finds specific documents, then combines all results!

```cs
var expander = new SubQueryExpansion(
llmEndpoint: "http://localhost:1234/v1",
llmApiKey: "your-key",
maxSubQueries: 4
);

var queries = expander.ExpandQuery(
"What is photosynthesis and how do plants use it for energy production?"
);
// Returns: ["What is photosynthesis?", "how do plants use it for energy production?", 
// "information about photosynthesis", "information about energy production"]
```

Why use SubQueryExpansion:

- Handles multi-part questions effectively
- Each sub-query is more precise (better matches)
- Covers all aspects of complex information needs
- Ideal for research questions, comprehensive queries

When NOT to use it:

- Simple, single-concept queries (unnecessary overhead)
- When you need documents covering ALL aspects together (decomposition loses connections)
- Very short queries (nothing to decompose)
- When retrieval latency is critical (multiple searches = slower)

## How It Works

SubQueryExpansion solves the "complex query problem" where a single query contains multiple information needs.
It intelligently detects complexity indicators (conjunctions, multiple questions, comma-separated concepts) and
decomposes the query into independent sub-queries that are easier to retrieve for. For example, "How does climate
change affect polar bears and what conservation efforts exist?" becomes two focused queries. This approach improves
both precision (each sub-query is more specific) and recall (broader topic coverage). The implementation uses
linguistic patterns to identify sub-queries but can integrate with LLMs for more sophisticated decomposition.
Results from all sub-queries are retrieved independently and merged to provide comprehensive coverage.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SubQueryExpansion(String,String,Int32)` | Initializes a new instance of the `SubQueryExpansion` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ExpandQuery(String)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `RegexTimeout` | Timeout for regex operations to prevent ReDoS attacks. |

