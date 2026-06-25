---
title: "GCDPLNetAlgorithm<T, TInput, TOutput>"
description: "Implementation of GCDPLNet: Graph-based Cross-Domain Prototype Learning Network."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of GCDPLNet: Graph-based Cross-Domain Prototype Learning Network.

## How It Works

GCDPLNet treats parameter groups as nodes in a graph and uses learned attention-based
message passing to propagate adaptation signals between groups. Each group computes a
gradient feature, then message passing allows information from related groups to influence
adaptation — enabling cross-domain knowledge transfer through parameter-space structure.

**Algorithm:**

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` |  |
| `MetaTrain(TaskBatch<,,>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_graphAttention` | Graph attention parameters: numNodes × numNodes (pairwise attention). |

