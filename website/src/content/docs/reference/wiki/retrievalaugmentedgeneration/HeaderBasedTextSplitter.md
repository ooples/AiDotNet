---
title: "HeaderBasedTextSplitter"
description: "Splits structured documents based on header tags (H1, H2, H3, etc.)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.ChunkingStrategies`

Splits structured documents based on header tags (H1, H2, H3, etc.).

## How It Works

Ideal for Markdown and HTML documents where headers provide natural semantic boundaries.
Preserves document structure and hierarchy.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HeaderBasedTextSplitter(Int32,Int32,Int32,Boolean)` | Initializes a new instance of the `HeaderBasedTextSplitter` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ChunkCore(String)` | Core chunking logic that splits text based on header hierarchy. |

