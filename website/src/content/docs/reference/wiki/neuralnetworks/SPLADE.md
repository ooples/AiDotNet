---
title: "SPLADE<T>"
description: "SPLADE (Sparse Lexical and Expansion Model) neural network implementation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

SPLADE (Sparse Lexical and Expansion Model) neural network implementation.
Maps text to a high-dimensional sparse vector in the vocabulary space.

## For Beginners

Imagine a dictionary with 30,000 words. For every sentence, SPLADE 
creates a giant list of 30,000 numbers, but almost all of them are zero. It only puts 
numbers next to the words that are actually important to the meaning.

## How It Works

SPLADE is a sparse retrieval model that learns to represent documents and queries as sparse 
vectors over the vocabulary. It uses a log-saturation effect and sparsity regularization 
(e.g., FLOPs or L1) to learn lexical expansion and term importance.

The "Expansion" part is the most interesting: if you say "The smartphone is fast," SPLADE 
might automatically put a number next to the word "Apple" or "Android" in its dictionary, 
even if you didn't say them. This helps it find relevant documents that use different words.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SPLADE` | Initializes a new instance of the SPLADE model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Embed(String)` | Encodes text into a high-dimensional sparse lexical representation. |
| `EmbedAsync(String)` |  |
| `EmbedBatchAsync(IEnumerable<String>)` |  |
| `GetModelMetadata` | Retrieves detailed metadata about the SPLADE configuration. |
| `GetOptions` |  |
| `InitializeLayers` | Configures the transformer backbone and the ReLU-based expansion head for SPLADE. |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |

