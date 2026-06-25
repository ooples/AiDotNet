---
title: "StopWordRemovalQueryProcessor"
description: "Removes common stop words from queries to improve retrieval precision."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.QueryProcessors`

Removes common stop words from queries to improve retrieval precision.

## For Beginners

Removes common words that don't help with searching.

Examples:

- "What are the main features of the new iPhone?"

→ "main features new iPhone"

- "How does a neural network learn from data?"

→ "neural network learn data"

Words like "what", "are", "the", "of" are removed because they don't help find specific documents.

## How It Works

This processor filters out common, uninformative words that don't contribute
to retrieval quality. By removing these words, the search becomes more focused
on meaningful content.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StopWordRemovalQueryProcessor(HashSet<String>,Boolean)` | Initializes a new instance of the StopWordRemovalQueryProcessor class. |

## Fields

| Field | Summary |
|:-----|:--------|
| `RegexTimeout` | Timeout for regex operations to prevent ReDoS attacks. |

