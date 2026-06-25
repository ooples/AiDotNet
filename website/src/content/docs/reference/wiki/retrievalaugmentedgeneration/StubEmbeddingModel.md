---
title: "StubEmbeddingModel<T>"
description: "A deterministic stub embedding model for testing and development that uses hash-based vector generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.Embeddings`

A deterministic stub embedding model for testing and development that uses hash-based vector generation.

## For Beginners

This is a simple placeholder until real embedding models are ready.

Think of it like using a stick figure placeholder in a drawing:

- It's not a real embedding model (like BERT or GPT)
- It converts text to numbers in a simple, predictable way (using hashing)
- Same text always gets same numbers (deterministic)
- Good enough for testing your RAG pipeline structure
- Replace it with a real embedding model for production

For example:

- "hello" always becomes the same vector
- "hello" and "world" get different vectors
- But similarity isn't semantically meaningful (unlike real embeddings)

This enables development work on Issue #284 without waiting for Issue #12.

## How It Works

This implementation creates deterministic vector embeddings based on text hashing.
It's designed for testing, development, and prototyping RAG systems before real embedding
models are available. The same input text always produces the same embedding vector,
making it suitable for unit tests and demonstrations.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StubEmbeddingModel(Int32,Int32)` | Initializes a new instance of the StubEmbeddingModel class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EmbeddingDimension` | Gets the dimensionality of the embedding vectors produced by this model. |
| `MaxTokens` | Gets the maximum length of text (in tokens) that this model can process. |

## Methods

| Method | Summary |
|:-----|:--------|
| `EmbedCore(String)` | Core embedding logic that generates a deterministic vector from text using hashing. |

