---
title: "LLMContextCompressor<T>"
description: "LLM-based context compression to reduce token usage while preserving key information."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.ContextCompression`

LLM-based context compression to reduce token usage while preserving key information.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LLMContextCompressor(Double,String,String)` | Initializes a new instance of the `LLMContextCompressor` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CompressCore(List<Document<>>,String,Dictionary<String,Object>)` | Compresses documents while preserving relevance to the query. |
| `CompressText(String,String)` | Compresses text based on relevance to the query. |

