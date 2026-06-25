---
title: "IChunkingStrategy"
description: "Defines the contract for text chunking strategies that split documents into smaller segments."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for text chunking strategies that split documents into smaller segments.

## For Beginners

A chunking strategy is like deciding how to slice a pizza.

Think of different ways to divide a long document:

- Fixed-size chunks: Cut every 500 words (like equal pizza slices)
- Sentence-based: Keep sentences together (like cutting between toppings)
- Paragraph-based: Keep paragraphs intact (like cutting by sections)
- Semantic: Group related content (like separating different flavor sections)

Why chunk documents?

- Long documents don't fit in the AI model (like a pizza too big for one plate)
- Smaller chunks make search more precise (finding exactly the relevant part)
- You can retrieve just the relevant sections, not entire documents

For example, searching a 100-page manual:

- Without chunking: Return the entire manual (overwhelming)
- With chunking: Return just the 2 paragraphs that answer your question (perfect!)

## How It Works

A chunking strategy determines how to divide large text documents into smaller, manageable pieces.
This is essential for RAG systems because embedding models have maximum token limits, and smaller
chunks enable more precise retrieval. Different strategies balance between preserving context
and creating appropriately-sized segments.

## Properties

| Property | Summary |
|:-----|:--------|
| `ChunkOverlap` | Gets the number of characters that should overlap between consecutive chunks. |
| `ChunkSize` | Gets the target size for each chunk in characters. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Chunk(String)` | Splits a text string into chunks according to the strategy's rules. |
| `ChunkWithPositions(String)` | Splits a text string into chunks and returns them with position metadata. |

