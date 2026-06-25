---
title: "ChunkingStrategyBase"
description: "Provides a base implementation for text chunking strategies with common functionality."
section: "API Reference"
---

`Base Classes` · `AiDotNet.RetrievalAugmentedGeneration.ChunkingStrategies`

Provides a base implementation for text chunking strategies with common functionality.

## For Beginners

This is the foundation that all text splitting methods build upon.

Think of it like a template for dividing text:

- It handles common tasks (checking inputs, managing overlap)
- Specific chunking methods (fixed-size, sentence-based) just fill in how they split text
- This ensures all chunking strategies work consistently

## How It Works

This abstract class implements the IChunkingStrategy interface and provides common functionality
for text splitting strategies. It handles validation and provides utility methods for chunk overlap
while allowing derived classes to focus on implementing the core chunking algorithm.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ChunkingStrategyBase(Int32,Int32)` | Initializes a new instance of the ChunkingStrategyBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ChunkOverlap` | Gets the number of characters that should overlap between consecutive chunks. |
| `ChunkSize` | Gets the target size for each chunk in characters. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Chunk(String)` | Splits a text string into chunks according to the strategy's rules. |
| `ChunkCore(String)` | Core chunking logic to be implemented by derived classes. |
| `ChunkWithPositions(String)` | Splits a text string into chunks and returns them with position metadata. |
| `CreateOverlappingChunks(String)` | Creates overlapping chunks from text using simple character-based splitting. |
| `SplitOnSentences(String,Char[])` | Splits text on sentence boundaries while respecting chunk size limits. |
| `ValidateText(String)` | Validates the input text. |

