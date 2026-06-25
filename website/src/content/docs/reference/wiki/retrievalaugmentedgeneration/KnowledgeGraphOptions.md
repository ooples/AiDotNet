---
title: "KnowledgeGraphOptions"
description: "Configuration options for advanced knowledge graph capabilities including embeddings, community detection, link prediction, temporal queries, and KG construction."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.RetrievalAugmentedGeneration.Graph`

Configuration options for advanced knowledge graph capabilities including embeddings,
community detection, link prediction, temporal queries, and KG construction.

## For Beginners

After setting up your knowledge graph via `ConfigureRetrievalAugmentedGeneration()`,
use `ConfigureKnowledgeGraph()` to enable advanced features:

- Train embeddings to enable link prediction and entity similarity
- Detect communities for global search capabilities
- Enable temporal queries for time-aware reasoning
- Construct a KG automatically from text input

## How It Works

These options are separate from `ConfigureRetrievalAugmentedGeneration()`, which handles
low-level plumbing (IGraphStore, KnowledgeGraph, HybridGraphRetriever). This class configures
higher-level features built on top of the existing infrastructure.

## Properties

| Property | Summary |
|:-----|:--------|
| `ConstructionOptions` | Options for KG construction from text. |
| `ConstructionTexts` | Text documents to construct the knowledge graph from. |
| `EmbeddingOptions` | Options for embedding model training. |
| `EmbeddingType` | Type of embedding model to use. |
| `EnableLinkPrediction` | Whether to enable link prediction. |
| `GraphRAGMode` | GraphRAG retrieval mode. |
| `GraphRAGOptions` | Options for GraphRAG retrieval. |
| `LinkPredictionMaxTestEdges` | Maximum number of test edges for link prediction evaluation. |
| `LinkPredictionTestFraction` | Fraction of edges to hold out for link prediction evaluation. |
| `TrainEmbeddings` | Whether to train knowledge graph embeddings. |

