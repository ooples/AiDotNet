---
title: "MatryoshkaEmbedding<T>"
description: "Matryoshka Representation Learning (MRL) neural network implementation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Matryoshka Representation Learning (MRL) neural network implementation.
Learns nested embeddings where smaller prefixes of the full vector are valid representations.

## For Beginners

Imagine a Russian nesting doll (a Matryoshka). Inside the big doll is a smaller one, 
and inside that is an even smaller one. MRL works the same way: it creates a long list of numbers 
to describe a sentence, but it makes sure that the first few numbers are a "perfect miniature" 
of the whole meaning. This lets you use a tiny list for a fast search and the full list when 
you need total accuracy.

## How It Works

Matryoshka Representation Learning (MRL) is a technique that enables a single model to adapt 
its embedding dimension to the requirements of the downstream task. It optimizes for multiple 
dimensions simultaneously, ensuring high accuracy even when using truncated vector prefixes.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MatryoshkaEmbedding` | Initializes a new instance with default architecture settings. |
| `MatryoshkaEmbedding(NeuralNetworkArchitecture<>,ITokenizer,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,Int32,Int32,Int32[],Int32,Int32,Int32,Int32,TransformerEmbeddingNetwork<>.PoolingStrategy,ILossFunction<>,Double,MatryoshkaEmbeddingOptions)` | Initializes a new instance of the MatryoshkaEmbedding model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Embed(String)` |  |
| `EmbedAsync(String)` |  |
| `EmbedBatchAsync(IEnumerable<String>)` |  |
| `EmbedResized(String,Int32)` | Encodes text into a truncated and re-normalized embedding of the requested dimension. |
| `GetModelMetadata` | Retrieves metadata about the Matryoshka configuration. |
| `GetOptions` |  |
| `InitializeLayers` | Configures the transformer encoder and projection layers for the MRL architecture. |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |

