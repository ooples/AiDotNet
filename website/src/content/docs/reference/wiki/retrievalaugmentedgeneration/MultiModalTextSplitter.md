---
title: "MultiModalTextSplitter"
description: "Multi-modal splitter for documents containing both text and images."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.ChunkingStrategies`

Multi-modal splitter for documents containing both text and images.

## How It Works

Creates chunks that keep text and related images together, preserving the relationship
between visual and textual content for better context preservation.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MultiModalTextSplitter(Int32,Int32,Int32,Boolean)` | Initializes a new instance of the `MultiModalTextSplitter` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ChunkCore(String)` | Core chunking logic that splits text while preserving text-image relationships. |

