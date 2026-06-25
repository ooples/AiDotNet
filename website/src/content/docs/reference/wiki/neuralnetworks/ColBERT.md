---
title: "ColBERT<T>"
description: "ColBERT (Contextualized Late Interaction over BERT) neural network implementation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

ColBERT (Contextualized Late Interaction over BERT) neural network implementation.
Uses token-level representations for high-precision document retrieval.

## For Beginners

Most AI search models are like people who read a whole book and then 
try to summarize it in just one word. ColBERT is like a person who keeps detailed notes 
on every single word. When you ask a question, ColBERT compares every word in your question 
to every word in the document notes. This is much more accurate because no information 
is "lost" during summarization.

## How It Works

ColBERT is a highly efficient and accurate retrieval model that keeps a separate vector for every 
token in a sentence. It calculates the similarity between a query and a document using a 
"Late Interaction" MaxSim operator, allowing it to capture fine-grained semantic matches.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ColBERT` | Initializes a new instance with default architecture settings. |
| `ColBERT(NeuralNetworkArchitecture<>,ITokenizer,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,Int32,Int32,Int32,Int32,Int32,Int32,ILossFunction<>,Double,ColBERTOptions)` | Initializes a new instance of the ColBERT model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Embed(String)` | Fallback method that encodes a sentence into a single summary vector (mean-pooled). |
| `EmbedLateInteraction(String)` | Encodes text into a multi-vector matrix where each row is a contextualized token embedding. |
| `GetModelMetadata` | Retrieves metadata about the ColBERT model. |
| `GetOptions` |  |
| `InitializeLayers` | Sets up the transformer layers and the token-level projection head for ColBERT. |
| `LateInteractionScore(Matrix<>,Matrix<>)` | Computes the similarity score between a query and document matrix using the MaxSim interaction. |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_outputDim` | The dimension of each token-level embedding vector (typically 128). |

