---
title: "SimCSE<T>"
description: "SimCSE (Simple Contrastive Learning of Sentence Embeddings) neural network implementation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

SimCSE (Simple Contrastive Learning of Sentence Embeddings) neural network implementation.

## For Beginners

Imagine you're trying to recognize a friend in a crowded room. Even if they 
are wearing a hat, glasses, or a scarf (like "dropout" noise), they are still the same person. 
SimCSE trains the model by showing it the same sentence twice with different "masks" and 
telling it: "this is the same sentence." This helps the model learn the true, deep meaning 
of the sentence that stays constant regardless of small changes.

## How It Works

SimCSE is a state-of-the-art framework for learning sentence embeddings. It uses a contrastive learning 
objective to pull semantically similar sentences together and push dissimilar ones apart. 
Its most famous variant is unsupervised, using different dropout masks on the same sentence 
as a minimal data augmentation.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SimCSE` | Initializes a new instance with default architecture settings. |
| `SimCSE(NeuralNetworkArchitecture<>,ITokenizer,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,SimCSEType,Int32,Int32,Int32,Int32,Int32,Int32,Double,TransformerEmbeddingNetwork<>.PoolingStrategy,ILossFunction<>,Double,SimCSEOptions)` | Initializes a new instance of the SimCSE model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Embed(String)` |  |
| `EmbedAsync(String)` |  |
| `EmbedBatchAsync(IEnumerable<String>)` |  |
| `GetModelMetadata` | Retrieves detailed metadata about the SimCSE configuration. |
| `GetOptions` |  |
| `InitializeLayers` | Configures the transformer encoder layers for SimCSE based on standard research patterns from LayerHelper. |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |

