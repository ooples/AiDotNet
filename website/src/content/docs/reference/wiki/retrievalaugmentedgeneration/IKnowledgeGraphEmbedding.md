---
title: "IKnowledgeGraphEmbedding<T>"
description: "Defines the contract for knowledge graph embedding models that learn vector representations of entities and relations."
section: "API Reference"
---

`Interfaces` · `AiDotNet.RetrievalAugmentedGeneration.Graph.Embeddings`

Defines the contract for knowledge graph embedding models that learn vector representations
of entities and relations.

## For Beginners

A knowledge graph stores facts as (head, relation, tail) triples,
e.g., (Einstein, born_in, Germany). Embedding models learn numeric vectors for each entity
and relation so that valid triples score higher than invalid ones. This lets you:

- Predict missing links: "What city was Tesla born in?"
- Find similar entities: Entities with similar vectors are semantically related
- Evaluate triple plausibility: Score how likely a new fact is to be true

## How It Works

Knowledge graph embeddings map entities and relations into a continuous vector space,
enabling operations like link prediction, entity clustering, and relation inference.

## Properties

| Property | Summary |
|:-----|:--------|
| `EmbeddingDimension` | Gets the dimensionality of the embedding vectors. |
| `IsDistanceBased` | Gets whether this model uses distance-based scoring (lower score = more plausible triple) or semantic matching scoring (higher score = more plausible triple). |
| `IsTrained` | Gets whether the model has been trained. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetEntityEmbedding(String)` | Gets the learned embedding vector for an entity. |
| `GetRelationEmbedding(String)` | Gets the learned embedding vector for a relation type. |
| `ScoreTriple(String,String,String)` | Scores a triple (head, relation, tail) — lower scores indicate more plausible triples for distance-based models, higher scores for semantic matching models. |
| `Train(KnowledgeGraph<>,KGEmbeddingOptions)` | Trains the embedding model on the triples from the given knowledge graph. |

