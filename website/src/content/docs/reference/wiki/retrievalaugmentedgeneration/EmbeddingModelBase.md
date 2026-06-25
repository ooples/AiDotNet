---
title: "EmbeddingModelBase<T>"
description: "Provides a base implementation for embedding models with common functionality."
section: "API Reference"
---

`Base Classes` · `AiDotNet.RetrievalAugmentedGeneration.Embeddings`

Provides a base implementation for embedding models with common functionality.

## For Beginners

This is the foundation that all embedding models build upon.

Think of it like a template for creating embedding models:

- It handles common tasks (checking inputs, batching text, normalizing vectors)
- Specific embedding models (BERT, GPT, etc.) just fill in how they convert text to numbers
- This keeps code consistent and reduces duplication

## How It Works

This abstract class implements the IEmbeddingModel interface and provides common functionality
for text embedding models. It handles validation, batching, and normalization while allowing
derived classes to focus on implementing the core embedding algorithm.

## Properties

| Property | Summary |
|:-----|:--------|
| `EmbeddingDimension` | Gets the dimensionality of the embedding vectors produced by this model. |
| `MaxTokens` | Gets the maximum length of text (in tokens) that this model can process. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateMatrixFromVectors(IList<Vector<>>)` | Creates a matrix from a collection of vectors. |
| `Dispose` | Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources. |
| `Dispose(Boolean)` | Releases unmanaged and - optionally - managed resources. |
| `Embed(String)` | Embeds a single text string into a vector representation. |
| `EmbedAsync(String)` | Asynchronously embeds a single text string into a vector representation. |
| `EmbedBatch(IEnumerable<String>)` | Embeds multiple text strings into vector representations in a single batch operation. |
| `EmbedBatchAsync(IEnumerable<String>)` | Asynchronously embeds multiple text strings into vector representations in a single batch operation. |
| `EmbedBatchCore(IList<String>)` | Core batch embedding logic to be implemented by derived classes. |
| `EmbedBatchCoreAsync(IList<String>)` | Asynchronous core batch embedding logic to be implemented by derived classes. |
| `GenerateFallbackEmbedding(String)` | Generates a deterministic fallback embedding based on the text hash. |
| `ValidateText(String)` | Validates the input text. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Gets the numeric operations for type T. |

