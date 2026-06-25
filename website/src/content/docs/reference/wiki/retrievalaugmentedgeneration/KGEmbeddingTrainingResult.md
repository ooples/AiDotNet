---
title: "KGEmbeddingTrainingResult"
description: "Contains the results of training a knowledge graph embedding model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.Graph.Embeddings`

Contains the results of training a knowledge graph embedding model.

## For Beginners

After training, this tells you:

- EpochLosses: How the error decreased over time (should go down)
- EntityCount/RelationCount: How many entities and relation types were learned
- TripleCount: Total number of facts used for training
- TrainingDuration: How long training took

## How It Works

This class captures training metrics including per-epoch loss values,
entity/relation counts, and training duration for diagnostics and evaluation.

## Properties

| Property | Summary |
|:-----|:--------|
| `EntityCount` | Number of unique entities in the training data. |
| `EpochLosses` | Average loss value for each training epoch. |
| `RelationCount` | Number of unique relation types in the training data. |
| `TotalEpochs` | Total number of epochs completed. |
| `TrainingDuration` | Wall-clock training duration. |
| `TripleCount` | Total number of triples used for training. |

