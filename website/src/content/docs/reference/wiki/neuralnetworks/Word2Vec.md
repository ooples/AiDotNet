---
title: "Word2Vec<T>"
description: "Word2Vec neural network implementation supporting both Skip-Gram and CBOW architectures."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Word2Vec neural network implementation supporting both Skip-Gram and CBOW architectures.

## For Beginners

Imagine you are learning a new language by looking at thousands of newspapers. 
You notice that the word "bark" often appears near "dog," "tree," and "loud." You also notice that 
"meow" appears near "cat," "kitten," and "soft." Word2Vec is an AI that does exactly this—it 
builds a "map" of words where words with similar meanings or contexts are placed close together. 
In this map, "dog" and "cat" might be neighbors, while "dog" and "spaceship" are on opposite ends.

## How It Works

Word2Vec is a foundational technique in Natural Language Processing (NLP) that learns to map words 
to dense vectors of real numbers. These "embeddings" capture semantic and syntactic relationships 
based on the contexts in which words appear.

This implementation supports two main styles:

- **Skip-Gram:** Tries to guess the surrounding "context" words when given a single target word.
- **CBOW:** Tries to guess a single "target" word when given a group of surrounding context words.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Word2Vec(NeuralNetworkArchitecture<>,ITokenizer,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,Int32,Int32,Int32,Int32,Word2VecType,ILossFunction<>,Double,Word2VecOptions)` | Initializes a new instance of the Word2Vec model. |

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
| `Embed(String)` | Encodes a single string into a normalized embedding vector by averaging its word vectors. |
| `EmbedAsync(String)` |  |
| `EmbedBatch(IEnumerable<String>)` | Encodes a collection of strings into a matrix of embedding vectors. |
| `EmbedBatchAsync(IEnumerable<String>)` |  |
| `Forward(Tensor<>)` | Performs a forward pass through the network, typically to retrieve an embedding for given token IDs. |
| `GetModelMetadata` | Retrieves detailed metadata about the Word2Vec model. |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the layers of the Word2Vec network based on the provided architecture or standard research defaults. |
| `PredictEager(Tensor<>)` | Routes inference through `Tensor{` for compiled-plan replay; `Tensor{` remains the eager fallback. |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` | Trains the Word2Vec model on a single batch of target and context pairs. |
| `UpdateParameters(List<Tensor<>>)` | Applies gradient clipping and uses the optimizer to update layer parameters. |
| `UpdateParameters(Vector<>)` | Updates the parameters of all layers in the network based on a provided update vector. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_embeddingDimension` | The length of the embedding vector for each word. |
| `_lossFunction` | The loss function used to measure how well the model is learning. |
| `_maxTokens` | The maximum number of tokens to process per input string. |
| `_optimizer` | The optimization algorithm used to update the model's parameters. |
| `_tokenizer` | The tokenizer used to convert text into numerical IDs. |
| `_type` | The specific Word2Vec architecture type (Skip-Gram or CBOW). |
| `_vocabSize` | The number of unique words in the model's vocabulary. |
| `_windowSize` | The size of the context window used during training. |

