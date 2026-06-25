---
title: "SpellCheckQueryProcessor"
description: "Processes queries by correcting common spelling errors using a dictionary-based approach."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.QueryProcessors`

Processes queries by correcting common spelling errors using a dictionary-based approach.

## For Beginners

This fixes spelling mistakes in your search queries.

Examples:

- "photsynthesis" → "photosynthesis"
- "artifical intelligence" → "artificial intelligence"
- "machin learning" → "machine learning"

It helps you find documents even when you make typos!

## How It Works

This processor improves retrieval accuracy by fixing typos and misspellings before
the query is sent to the retriever. Uses a simple edit distance algorithm combined
with a frequency-based dictionary to suggest corrections.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SpellCheckQueryProcessor(Dictionary<String,String>,Int32)` | Initializes a new instance of the SpellCheckQueryProcessor class. |

