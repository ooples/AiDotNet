---
title: "AutoCompressor<T>"
description: "Auto-compressor using rule-based text compression for document content reduction."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.ContextCompression`

Auto-compressor using rule-based text compression for document content reduction.

## How It Works

Compresses documents by extracting the most relevant sentences based on keyword importance
and position in the document. This is a production implementation that doesn't require
external ML models.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AutoCompressor(Int32,Double)` | Initializes a new instance of the `AutoCompressor` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CompressCore(List<Document<>>,String,Dictionary<String,Object>)` | Compresses documents using rule-based sentence extraction. |

