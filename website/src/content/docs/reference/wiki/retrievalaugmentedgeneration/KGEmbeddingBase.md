---
title: "KGEmbeddingBase<T>"
description: "Abstract base class for knowledge graph embedding models providing shared training infrastructure."
section: "API Reference"
---

`Base Classes` · `AiDotNet.RetrievalAugmentedGeneration.Graph.Embeddings`

Abstract base class for knowledge graph embedding models providing shared training infrastructure.

## For Beginners

This class handles the "plumbing" of training:

1. Builds entity/relation vocabularies from the graph
2. Initializes random embedding vectors
3. For each epoch, shuffles triples into mini-batches
4. For each positive triple, generates corrupted (negative) triples
5. Computes loss and gradients, then updates embeddings via SGD
6. Subclasses define how to score triples and compute gradients

## How It Works

This base class implements the common training loop with mini-batch SGD and negative sampling.
Subclasses only need to implement scoring, gradient computation, and post-epoch normalization.

## Properties

| Property | Summary |
|:-----|:--------|
| `EmbeddingDimension` |  |
| `IsDistanceBased` |  |
| `IsTrained` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeLossAndUpdateGradients(Int32,Int32,Int32,Int32,Int32,Double,KGEmbeddingOptions)` | Computes loss for a positive triple vs. |
| `GetEntityEmbedding(String)` |  |
| `GetEntityEmbeddingSize` | Gets the size of entity embedding vectors. |
| `GetRelationEmbedding(String)` |  |
| `GetRelationEmbeddingSize` | Gets the size of relation embedding vectors. |
| `OnInitialize(KGEmbeddingOptions,Random,KnowledgeGraph<>)` | Called after embedding arrays are allocated but before the training loop begins. |
| `OnPostEpoch(Int32)` | Called after each epoch. |
| `ScoreTriple(String,String,String)` |  |
| `ScoreTripleInternal(Int32,Int32,Int32)` | Computes the score for a triple given entity/relation indices. |
| `Train(KnowledgeGraph<>,KGEmbeddingOptions)` |  |

