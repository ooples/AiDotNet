---
title: "GloVe<T>"
description: "GloVe (Global Vectors for Word Representation) neural network implementation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

GloVe (Global Vectors for Word Representation) neural network implementation.

## For Beginners

If Word2Vec is like a student learning from reading newspapers one page at a time, 
GloVe is like a researcher who looks at the entire library all at once. It builds a giant table 
showing how often every word in the dictionary appears near every other word. It then uses 
clever math to find the best "address" for each word so that the distance between addresses 
matches those counts perfectly.

## How It Works

GloVe is an unsupervised learning algorithm for obtaining vector representations for words. 
Training is performed on aggregated global word-word co-occurrence statistics from a corpus, 
and the resulting representations showcase interesting linear substructures of the word vector space.

The GloVe model is famous for its ability to solve word analogies, like: 
"King - Man + Woman = Queen."

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GloVe` | Initializes a new instance of the GloVe embedding model. |
| `GloVe(NeuralNetworkArchitecture<>,ITokenizer,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,Int32,Int32,Int32,ILossFunction<>,Double,GloVeOptions)` | Initializes a new instance of the GloVe model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EmbeddingDimension` |  |
| `MaxTokens` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateBertCompatible` | Creates a GloVe instance with the legacy 768-dim default that this project's parameterless constructor used before the paper-faithful `d = 100` switch. |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Embed(String)` | Turns a sentence into a single, summary coordinate (embedding). |
| `EmbedAsync(String)` |  |
| `EmbedBatch(IEnumerable<String>)` | Encodes a whole batch of sentences at once for speed. |
| `EmbedBatchAsync(IEnumerable<String>)` |  |
| `Forward(Tensor<>)` | Performs a forward pass to retrieve embeddings for given token IDs. |
| `ForwardForTraining(Tensor<>)` |  |
| `GetModelMetadata` | Returns technical details and configuration info about the GloVe model. |
| `GetOptions` |  |
| `InitializeLayers` | Sets up the neural network layers required for the GloVe architecture. |
| `PredictEager(Tensor<>)` | Routes inference through `Tensor{` for compiled-plan replay; `Tensor{` remains the eager fallback. |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` | Trains the model on a batch of word pairs and their co-occurrence counts. |
| `UpdateParameters(List<Tensor<>>)` | Clips gradients and uses the optimizer to update layer parameters. |
| `UpdateParameters(Vector<>)` | Updates the internal weights and biases of the model. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_embeddingDimension` | The number of dimensions in the learned word vector space. |
| `_fallbackTokenizer` | Cached fallback tokenizer to avoid per-call creation. |
| `_lossFunction` | The loss function used to evaluate training progress. |
| `_maxTokens` | The maximum number of tokens to process per input. |
| `_optimizer` | The optimization algorithm used to refine the word vectors. |
| `_tokenizer` | The tokenizer used to map text strings to numerical IDs. |
| `_vocabSize` | The number of unique words the model is capable of representing. |

