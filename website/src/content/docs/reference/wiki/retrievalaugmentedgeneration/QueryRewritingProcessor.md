---
title: "QueryRewritingProcessor<T>"
description: "Rewrites queries for clarity and completeness, especially in conversational contexts."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.QueryProcessors`

Rewrites queries for clarity and completeness, especially in conversational contexts.

## For Beginners

Makes incomplete questions complete by adding missing context.

Conversational Examples:

- User: "Tell me about transformers"
- User: "What about their applications?" → "What are the applications of transformers?"

Clarity Examples:

- "how r cars made" → "how are cars manufactured"
- "wht is AI" → "what is artificial intelligence"

This makes your searches clearer and gets better results!

## How It Works

This processor transforms conversational or context-dependent queries into standalone,
clear questions. It's particularly useful in multi-turn conversations where queries
reference previous context.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `QueryRewritingProcessor(IGenerator<>)` | Initializes a new instance of the QueryRewritingProcessor class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddToHistory(String)` | Adds a query to the conversation history for context-aware rewriting. |

