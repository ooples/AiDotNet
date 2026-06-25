---
title: "TaskCondHyperNetOptions<T, TInput, TOutput>"
description: "Configuration options for Task-Conditioned HyperNetwork."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for Task-Conditioned HyperNetwork.

## How It Works

A hypernetwork generates task-specific parameter deltas conditioned on a task embedding
derived from support-set gradient statistics. The hypernetwork is a 2-layer MLP:
embedding → hidden → parameter delta, added to the meta-initialization.

## Properties

| Property | Summary |
|:-----|:--------|
| `ChunkSize` | Chunk size for chunked hypernetwork output (generates parameters in chunks to reduce hypernetwork size). |
| `EmbeddingDim` | Dimension of the task embedding derived from gradient statistics. |
| `EmbeddingRegWeight` | L2 regularization weight on task embedding magnitude. |
| `HyperHiddenDim` | Hidden dimension of the hypernetwork MLP. |

