---
title: "AgenticChunker"
description: "Production-ready intelligent chunker that decides where to split text based on semantic boundaries."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.ChunkingStrategies`

Production-ready intelligent chunker that decides where to split text based on semantic boundaries.

## For Beginners

This is like a smart text splitter that understands content structure.

Think of it like organizing a book:

- Don't split in the middle of a sentence or paragraph
- Keep related ideas together in the same chunk
- Start new chunks at natural topic boundaries
- Maintain context with overlapping content

How it works:

1. Identifies structural elements (paragraphs, sections, lists)
2. Calculates semantic coherence scores for potential splits
3. Creates chunks at natural boundaries
4. Adds overlap for context preservation

Example:

- Input: Long article about climate change
- Output: Chunks at section boundaries, keeping introduction separate from data analysis,

solutions separate from problems, etc.

Unlike simple fixed-size chunking:

- ✓ Respects paragraph boundaries
- ✓ Keeps related sentences together 
- ✓ Detects topic changes
- ✓ Preserves document structure

Production features:

- No external API dependencies
- Fast heuristic-based topic detection
- Configurable chunk sizes and overlap
- Handles multiple document formats
- Maintains semantic coherence

## How It Works

This chunker analyzes text structure to identify optimal split points based on:

- Paragraph boundaries
- Topic transitions (detected via sentence similarity)
- Natural breaks in content flow (headers, lists, code blocks)
- Semantic coherence within chunks

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AgenticChunker(Int32,Int32,Double)` | Initializes a new instance of the `AgenticChunker` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateSemanticCoherence(String,String)` | Calculates semantic coherence between two text segments using lexical overlap and connectivity. |
| `ChunkCore(String)` | Splits text into chunks using intelligent boundary detection. |

