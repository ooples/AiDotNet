---
title: "IEmbeddingModel<T>"
description: "Defines the contract for embedding models that convert text into vector representations."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for embedding models that convert text into vector representations.

## For Beginners

An embedding model is like a translator that converts words into numbers.

Think of it like a coordinate system for meaning:

- Each word or sentence becomes a point in high-dimensional space
- Similar meanings end up close together (like "happy" near "joyful")
- Different meanings are far apart (like "happy" far from "sad")
- This lets computers understand and compare text by measuring distances

For example, the embedding for "cat" might be close to "kitten" and "feline",
but far from "democracy" or "algorithm".

## How It Works

An embedding model transforms text into dense numerical vectors that capture semantic meaning.
These vectors enable similarity comparisons and are fundamental to retrieval-augmented generation.
The interface supports both single and batch embeddings with configurable dimensions.

## Properties

| Property | Summary |
|:-----|:--------|
| `EmbeddingDimension` | Gets the dimensionality of the embedding vectors produced by this model. |
| `MaxTokens` | Gets the maximum length of text (in tokens) that this model can process. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Embed(String)` | Embeds a single text string into a vector representation. |
| `EmbedAsync(String)` | Asynchronously embeds a single text string into a vector representation. |
| `EmbedBatch(IEnumerable<String>)` | Embeds multiple text strings into vector representations in a single batch operation. |
| `EmbedBatchAsync(IEnumerable<String>)` | Asynchronously embeds multiple text strings into vector representations in a single batch operation. |

