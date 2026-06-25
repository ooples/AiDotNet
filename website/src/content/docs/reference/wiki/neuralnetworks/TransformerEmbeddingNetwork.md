---
title: "TransformerEmbeddingNetwork<T>"
description: "A customizable Transformer-based embedding network."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

A customizable Transformer-based embedding network.
This serves as the high-performance foundation for modern sentence and document encoders.

## For Beginners

This is a "universal reading brain." Transformers are the most powerful 
type of AI for understanding language because they can look at every word in a sentence 
at the same time and see how they all relate. This customizable version lets you decide 
how many layers of thinking the brain should have, and how it should summarize its 
thoughts into a final list of numbers (the embedding).

## How It Works

This network provides a flexible implementation of the Transformer encoder architecture, 
enabling the generation of high-quality semantic embeddings. It supports multiple 
pooling strategies (Mean, Max, ClsToken) to aggregate token-level information.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TransformerEmbeddingNetwork` | Initializes a new instance with default architecture settings. |
| `TransformerEmbeddingNetwork(NeuralNetworkArchitecture<>,ITokenizer,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,Int32,Int32,Int32,Int32,Int32,Int32,TransformerEmbeddingNetwork<>.PoolingStrategy,ILossFunction<>,Double,TransformerEmbeddingOptions)` | Initializes a new instance of the TransformerEmbeddingNetwork. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EmbeddingDimension` |  |
| `MaxTokens` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Embed(String)` | Encodes a single string into a normalized summary vector. |
| `EmbedAsync(String)` |  |
| `EmbedBatch(IEnumerable<String>)` | Encodes a collection of strings into a matrix of embeddings. |
| `EmbedBatchAsync(IEnumerable<String>)` |  |
| `GetModelMetadata` | Returns metadata about the transformer network configuration. |
| `GetOptions` |  |
| `InitializeLayers` | Sets up the layer stack for the transformer network, including embedding, positional encoding, and transformer blocks. |
| `PoolOutput(Tensor<>)` | Applies the configured pooling strategy to convert token-level outputs into a sentence representation. |
| `PredictCore(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` | Trains the transformer model on a single batch of data. |
| `UpdateParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_embeddingDimension` | The size of the output embedding vector. |
| `_fallbackTokenizer` | Cached fallback tokenizer to avoid per-call creation. |
| `_feedForwardDim` | The dimensionality of the internal feed-forward hidden layers. |
| `_lossFunction` | The loss function used during training. |
| `_maxSequenceLength` | The maximum sequence length allowed for input text. |
| `_numHeads` | The number of attention heads used in each multi-head attention layer. |
| `_numLayers` | The total number of transformer encoder layers in the stack. |
| `_optimizer` | The optimization algorithm used to refine the network's parameters. |
| `_poolingStrategy` | The strategy used to pool token-level representations into a single vector. |
| `_tokenizer` | The tokenizer used to translate raw text into numerical token IDs. |
| `_vocabSize` | The size of the model's vocabulary (number of unique tokens). |

