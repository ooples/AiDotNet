---
title: "SentenceChunkingStrategy"
description: "Splits text into chunks based on sentence boundaries to preserve semantic coherence."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.ChunkingStrategies`

Splits text into chunks based on sentence boundaries to preserve semantic coherence.

## For Beginners

This keeps complete sentences together in each chunk.

Think of it like organizing a book:

- Bad way: Cut every 500 characters, even mid-sentence

"The cat sat on the m|at. The dog ran ar|ound the yard."

- Good way: Keep sentences whole

Chunk 1: "The cat sat on the mat. The dog ran around the yard."
Chunk 2: "The bird flew over the fence. The fish swam in the pond."

Why this matters:

- Retrieval works better when searching complete thoughts
- Generators get more coherent context
- No weird sentence fragments that confuse the model

Parameters:

- targetChunkSize: Aim for this many characters per chunk
- maxChunkSize: Never exceed this size (may break sentences if needed)
- overlapSentences: Number of sentences to repeat between chunks for context

Example with targetChunkSize=100, overlapSentences=1:
"First sentence. Second sentence. Third sentence. Fourth sentence."

Chunk 1: "First sentence. Second sentence. Third sentence."
Chunk 2: "Third sentence. Fourth sentence." (overlap: "Third sentence")

## How It Works

This chunking strategy splits text at sentence boundaries (periods, question marks,
exclamation points) and combines sentences until reaching the target chunk size.
This approach preserves complete thoughts and improves retrieval quality compared
to arbitrary character-based splitting.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SentenceChunkingStrategy(Int32,Int32,Int32)` | Initializes a new instance of the SentenceChunkingStrategy class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ChunkCore(String)` | Splits text into chunks at sentence boundaries with accurate position tracking. |
| `SplitIntoSentencesWithPositions(String)` | Splits text into individual sentences with their positions in the original text. |

