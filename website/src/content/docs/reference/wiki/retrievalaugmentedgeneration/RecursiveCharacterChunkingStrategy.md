---
title: "RecursiveCharacterChunkingStrategy"
description: "Recursively splits text using a hierarchy of separators to preserve document structure."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.ChunkingStrategies`

Recursively splits text using a hierarchy of separators to preserve document structure.

## For Beginners

This is a smart splitter that keeps related text together.

Think of it like organizing a document by trying the best splits first:

Priority 1: Split by double newlines (paragraphs)
"Paragraph 1\n\nParagraph 2\n\nParagraph 3"
→ Keeps each paragraph whole

Priority 2: If paragraphs are too big, split by single newlines (sentences/lines)
"Long paragraph with\nmultiple lines\nthat need splitting"
→ Splits at line breaks

Priority 3: If lines are too big, split by periods (sentences)
"First sentence. Second sentence. Third sentence."
→ Splits at sentences

Priority 4: If sentences are too big, split by spaces (words)
"This is a very long sentence without periods"
→ Splits at words

Priority 5: Last resort, split by characters
"ReallyLongWordWithNoSpaces"
→ Splits anywhere

Why this is better than simple splitting:

- Keeps paragraphs together when possible (best semantic unity)
- Falls back gracefully when content is too large
- Preserves natural document structure
- Works well with various document formats (code, articles, books)

Example with chunkSize=100, overlap=20:

Input: "First paragraph.\n\nSecond paragraph that is very long and needs to be split into multiple chunks.\n\nThird paragraph."

1. Try splitting by "\n\n" → Second paragraph too large
2. Split second paragraph by " " → Gets multiple chunks
3. Add overlap between chunks

Result:

- Chunk 1: "First paragraph."
- Chunk 2: "Second paragraph that is very long and" (overlap from chunk 1)
- Chunk 3: "very long and needs to be split into" (overlap from chunk 2)
- Chunk 4: "split into multiple chunks."
- Chunk 5: "Third paragraph."

## How It Works

This advanced chunking strategy tries to split text using the most semantically meaningful
separators first (e.g., double newlines for paragraphs), falling back to less meaningful
separators (single newlines, spaces) only when necessary. This preserves the natural
structure of documents and keeps related content together.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RecursiveCharacterChunkingStrategy(Int32,Int32,String[])` | Initializes a new instance of the RecursiveCharacterChunkingStrategy class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ChunkCore(String)` | Recursively splits text using the separator hierarchy. |
| `SplitTextRecursively(String,String[])` | Recursively splits text, trying each separator in order. |

