---
title: "KeywordExtractionQueryProcessor"
description: "Extracts key terms and phrases from queries for focused retrieval."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.QueryProcessors`

Extracts key terms and phrases from queries for focused retrieval.

## For Beginners

Picks out the important words from your question.

Examples:

- "Explain the principles of quantum entanglement in simple terms"

→ "quantum entanglement principles simple terms"

- "What are the main features of the new iPhone?"

→ "main features new iPhone"

- "Can you tell me about machine learning algorithms?"

→ "machine learning algorithms"

This focuses your search on what really matters!

## How It Works

This processor identifies and extracts the most important keywords from a query,
removing filler words and focusing on content-bearing terms. This helps create
more focused and efficient searches.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `KeywordExtractionQueryProcessor(HashSet<String>,Int32)` | Initializes a new instance of the KeywordExtractionQueryProcessor class. |

## Fields

| Field | Summary |
|:-----|:--------|
| `RegexTimeout` | Timeout for regex operations to prevent ReDoS attacks. |

