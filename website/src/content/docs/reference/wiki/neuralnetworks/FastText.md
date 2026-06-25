---
title: "FastText<T>"
description: "FastText neural network implementation, an extension of Word2Vec that considers subword information."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

FastText neural network implementation, an extension of Word2Vec that considers subword information.

## For Beginners

Most models see words like "playing" and "played" as completely different things. 
FastText is smarter: it breaks words into pieces (like "play", "ing", and "ed"). Because it knows what 
"play" means, it can guess the meaning of a new word like "player" even if it has never seen it before. 
It's like a person who can understand a complex new word by looking at its root and its suffix.

## How It Works

FastText is a library for learning of word representations and sentence classification. It improves 
upon the original Word2Vec by representing each word as a bag of character n-grams. This approach 
allows the model to compute word representations for words that did not appear in the training data 
(out-of-vocabulary words).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FastText` | Initializes a new instance with default architecture settings. |
| `FastText(NeuralNetworkArchitecture<>,ITokenizer,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,Int32,Int32,Int32,Int32,ILossFunction<>,Double,FastTextOptions)` | Initializes a new instance of the FastText model. |

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
| `Embed(String)` | Turns text into a robust embedding vector using both word and subword information. |
| `EmbedAsync(String)` |  |
| `EmbedBatch(IEnumerable<String>)` | Encodes a batch of texts for high-throughput processing. |
| `EmbedBatchAsync(IEnumerable<String>)` |  |
| `Forward(Tensor<>)` | Performs a forward pass to retrieve representations. |
| `GetModelMetadata` | Retrieves detailed metadata about the FastText model configuration. |
| `GetOptions` |  |
| `InitializeLayers` | Configures the layers needed for FastText, including word and subword embedding tables. |
| `PredictEager(Tensor<>)` | Routes inference through `Tensor{` for compiled-plan replay; `Tensor{` remains the eager fallback. |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` | Trains the model on a single step of data using standard backpropagation. |
| `UpdateParameters(Vector<>)` | Updates all trainable weights in the FastText model. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_bucketSize` | The number of "buckets" used to store subword (n-gram) information. |
| `_embeddingDimension` | The dimensionality of the embedding vectors. |
| `_lossFunction` | The loss function used during training. |
| `_maxTokens` | The maximum number of tokens to process per input string. |
| `_optimizer` | The optimizer used to update the model's parameters. |
| `_tokenizer` | The tokenizer used to process text input. |
| `_vocabSize` | The size of the full-word vocabulary. |

