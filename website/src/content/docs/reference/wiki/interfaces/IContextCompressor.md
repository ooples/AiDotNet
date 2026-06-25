---
title: "IContextCompressor<T>"
description: "Defines the contract for compressing context documents to reduce token usage while preserving relevance."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for compressing context documents to reduce token usage while preserving relevance.

## For Beginners

A context compressor is like a summarizer for search results.

Think of it like preparing a briefing:

- You retrieve 10 long documents (might be 50,000 tokens)
- Your language model can only handle 8,000 tokens
- The compressor extracts key sentences and information
- Result: Compressed to 5,000 tokens with the most important content

This ensures you can:

- Use more retrieved documents without hitting token limits
- Focus on the most relevant parts of each document
- Reduce costs by using fewer tokens
- Still get accurate answers from the compressed context

## How It Works

A context compressor reduces the size of retrieved documents before passing them to a language model,
helping to stay within token limits while maintaining the most relevant information.

## Methods

| Method | Summary |
|:-----|:--------|
| `Compress(List<Document<>>,String,Dictionary<String,Object>)` | Compresses a collection of documents while preserving relevance to the query. |

