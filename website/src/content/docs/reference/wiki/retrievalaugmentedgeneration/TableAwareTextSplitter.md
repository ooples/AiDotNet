---
title: "TableAwareTextSplitter"
description: "Specialized splitter that correctly parses and chunks tabular data from documents."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.ChunkingStrategies`

Specialized splitter that correctly parses and chunks tabular data from documents.

## How It Works

Handles various table formats (Markdown, CSV, HTML tables) and ensures table integrity
by keeping related rows together and preserving column headers.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TableAwareTextSplitter(Int32,Int32,Int32,Boolean)` | Initializes a new instance of the `TableAwareTextSplitter` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ChunkCore(String)` | Splits text while preserving table structure. |

