---
title: "SlidingWindowChunkingStrategy"
description: "Sliding window chunking strategy with configurable window size and stride."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.ChunkingStrategies`

Sliding window chunking strategy with configurable window size and stride.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SlidingWindowChunkingStrategy(Int32,Int32)` | Initializes a new instance of the `SlidingWindowChunkingStrategy` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ChunkCore(String)` | Chunks text using a sliding window approach. |
| `ValidateAndCalculateOverlap(Int32,Int32)` | Validates parameters and calculates overlap before base constructor is called. |

