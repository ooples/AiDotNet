---
title: "SGPT<T>"
description: "SGPT (Sentence GPT) neural network implementation using decoder-only transformer architectures."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

SGPT (Sentence GPT) neural network implementation using decoder-only transformer architectures.

## For Beginners

Most AI models are like "readers" who read a whole sentence and then think about it. 
SGPT is like a "writer." Because it's trained to write sentences one word at a time, it has a 
very deep understanding of how sentences are built. When it finishes a sentence, the very 
last word it would have written contains a "mental summary" of everything that came before it. 
SGPT uses that summary as the coordinate (embedding) for the whole sentence.

## How It Works

SGPT leverages large-scale decoder-only models (like GPT-2 or GPT-Neo) to generate high-quality 
sentence embeddings. By focusing on the last token of a sequence, the model utilizes the 
unidirectional context to summarize the entire sentence's meaning.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SGPT` | Initializes a new instance of the SGPT model. |
| `SGPT(NeuralNetworkArchitecture<>,ITokenizer,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,Int32,Int32,Int32,Int32,Int32,Int32,TransformerEmbeddingNetwork<>.PoolingStrategy,ILossFunction<>,Double,SGPTOptions)` | Initializes a new instance of the SGPT model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Embed(String)` |  |
| `EmbedAsync(String)` |  |
| `EmbedBatchAsync(IEnumerable<String>)` |  |
| `GetModelMetadata` | Retrieves metadata about the SGPT model. |
| `GetOptions` |  |
| `InitializeLayers` | Configures the transformer layers for the SGPT model using decoder-only defaults from LayerHelper. |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |

