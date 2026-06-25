---
title: "SelectiveContextCompressor<T>"
description: "Selective context compressor that picks the most relevant sentences based on the query."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.ContextCompression`

Selective context compressor that picks the most relevant sentences based on the query.

## How It Works

Analyzes retrieved documents and selectively extracts only the sentences most relevant
to the query, reducing context length while preserving important information.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SelectiveContextCompressor(Int32,)` | Initializes a new instance of the `SelectiveContextCompressor` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CompressCore(List<Document<>>,String,Dictionary<String,Object>)` | Compresses documents by selecting relevant sentences. |

