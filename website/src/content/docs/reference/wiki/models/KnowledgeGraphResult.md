---
title: "KnowledgeGraphResult<T>"
description: "Contains the results of knowledge graph processing, including trained embeddings, community structure, and link prediction evaluation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models.Results`

Contains the results of knowledge graph processing, including trained embeddings,
community structure, and link prediction evaluation.

## For Beginners

After building a model with `ConfigureKnowledgeGraph()`,
this result contains everything that was computed:

- EmbeddingTrainingResult: Training statistics if embeddings were trained
- TrainedEmbedding: The trained embedding model for scoring triples
- CommunityStructure: Detected communities if Leiden was run
- CommunitySummaries: Human-readable descriptions of each community
- LinkPredictionEvaluation: Quality metrics if link prediction was evaluated
- EnhancedGraphRAG: The configured GraphRAG instance for querying

## Properties

| Property | Summary |
|:-----|:--------|
| `CommunityStructure` | Community detection results from the Leiden algorithm, if community detection was run. |
| `CommunitySummaries` | Human-readable summaries of detected communities. |
| `EmbeddingTrainingResult` | Training result from embedding model training, if embeddings were trained. |
| `EnhancedGraphRAG` | The configured EnhancedGraphRAG instance for querying. |
| `LinkPredictionEvaluation` | Link prediction evaluation metrics, if evaluation was performed. |
| `TrainedEmbedding` | The trained embedding model, if training was performed. |

