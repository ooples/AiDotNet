---
title: "SiameseNeuralNetwork<T>"
description: "Sentence-BERT (SBERT) style shared sentence-encoder tower: a transformer encoder that maps a tokenized input to a fixed-size embedding (default 768-d, BERT vocab 30522, max length 512) for semantic similarity and retrieval."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Sentence-BERT (SBERT) style shared sentence-encoder tower: a transformer encoder
that maps a tokenized input to a fixed-size embedding (default 768-d, BERT vocab
30522, max length 512) for semantic similarity and retrieval.

## For Beginners

A Siamese Network is like having two identical twins who think exactly 
the same way. You give a different photo to each twin, and they each describe what they see 
using a list of numbers. Because the twins think the same way, if the photos are similar, 
their descriptions will be almost identical. This is the most popular way to build face 
recognition or "find similar" search systems.

## How It Works

This implements the SHARED ENCODER of a Siamese/dual-encoder setup (Reimers & Gurevych
2019, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"): the same encoder
is applied independently to each sentence, and the resulting embeddings are compared by a
distance/cosine metric. `Predict` returns the encoder embedding for one input;
pair training (contrastive/triplet) compares two such embeddings.

NOTE: this is distinct from `SiameseNetwork`, which is the
pair-in / similarity-score-out VERIFICATION network of Koch et al. 2015 (twin subnetwork +
L1-distance sigmoid head). This class is the sentence-embedding ENCODER tower; that class
is the end-to-end pair verifier.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SiameseNeuralNetwork` | Initializes a new instance of the SiameseNeuralNetwork with default configuration. |
| `SiameseNeuralNetwork(NeuralNetworkArchitecture<>,ITokenizer,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,Int32,Int32,Int32,ILossFunction<>,Double,SiameseNeuralNetworkOptions)` | Initializes a new instance of the SiameseNeuralNetwork model. |

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
| `Embed(String)` | Encodes a single string into a normalized embedding vector using the shared encoder brain. |
| `EmbedAsync(String)` | Asynchronously encodes a single string into a normalized embedding vector. |
| `EmbedBatch(IEnumerable<String>)` | Encodes a batch of strings into a matrix of embedding vectors. |
| `EmbedBatchAsync(IEnumerable<String>)` | Asynchronously encodes a batch of strings into a matrix of embedding vectors. |
| `Forward(Tensor<>)` | Performs a forward pass through the shared encoder. |
| `GetModelMetadata` | Retrieves metadata about the Siamese dual-encoder model. |
| `GetOptions` |  |
| `InitializeLayers` | Sets up the shared encoder layers for the Siamese twins using defaults from LayerHelper. |
| `PredictEager(Tensor<>)` | Routes inference through `Tensor{` for compiled-plan replay; `Tensor{` remains the eager fallback. |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` | Trains the model on pairs of inputs using a similarity learning objective. |
| `UpdateParameters(Vector<>)` | Updates the shared parameters of the dual encoders. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_embeddingDimension` | The dimensionality of the shared embedding space. |
| `_lossFunction` | The loss function used to evaluate similarity (defaults to ContrastiveLoss). |
| `_maxSequenceLength` | The maximum length of input sequences the model will process. |
| `_optimizer` | The optimization algorithm used to update the shared parameters of the dual encoders. |
| `_tokenizer` | The tokenizer used to process text inputs into numerical token IDs. |
| `_vocabSize` | The number of unique tokens the shared encoder can recognize. |

