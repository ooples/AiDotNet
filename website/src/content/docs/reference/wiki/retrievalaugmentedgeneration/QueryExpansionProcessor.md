---
title: "QueryExpansionProcessor"
description: "Expands queries with synonyms and related terms to improve retrieval recall."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.QueryProcessors`

Expands queries with synonyms and related terms to improve retrieval recall.

## For Beginners

Adds related words to your search to find more results.

Examples:

- "AI models" → "AI models artificial intelligence machine learning models"
- "car" → "car automobile vehicle transportation"
- "photo" → "photo image picture photograph"

This helps you find documents even when they use different words for the same concept!

## How It Works

This processor broadens the search by adding semantically similar terms to the original query.
This helps retrieve relevant documents that might use different terminology.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `QueryExpansionProcessor(Dictionary<String,String[]>,Boolean)` | Initializes a new instance of the QueryExpansionProcessor class. |

