---
title: "DPGNAlgorithm<T, TInput, TOutput>"
description: "Implementation of DPGN (Distribution Propagation Graph Network) (Yang et al., CVPR 2020)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of DPGN (Distribution Propagation Graph Network) (Yang et al., CVPR 2020).

## For Beginners

DPGN uses TWO graphs working together:

**Graph 1 - Point Graph:**

- Nodes = examples (support + query)
- Edges = feature similarity
- Message passing: Share feature information between similar examples
- Result: Refined, context-aware features

**Graph 2 - Distribution Graph:**

- Same nodes but edges = distribution similarity
- Passes uncertainty/confidence information
- Result: Each node knows how certain it is

**Why two graphs?**
Knowing features isn't enough. You also need to know CONFIDENCE.
Two examples might have similar features but very different confidences.
The distribution graph captures this distinction.

**How they interact:**
After each propagation layer:

1. Point graph updates features (making them more discriminative)
2. Distribution graph updates confidences (making uncertainty estimates better)
3. Both use each other's output as input for the next layer

## How It Works

DPGN constructs dual graphs over support and query examples: a point graph for feature
propagation and a distribution graph for uncertainty propagation. Both graphs are refined
through multiple layers of message passing.

**Algorithm - DPGN:**

Reference: Yang, L., Li, L., Zhang, Z., Zhou, X., Zhou, E., & Liu, Y. (2020).
DPGN: Distribution Propagation Graph Network for Few-Shot Learning. CVPR 2020.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DPGNAlgorithm(DPGNOptions<,,>)` | Initializes a new DPGN meta-learner. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` |  |
| `ComputeAuxLoss(TaskBatch<,,>)` | Computes the average loss over a task batch using dual graph propagation. |
| `DualGraphPropagate(Vector<>)` | Performs multi-layer dual graph propagation: alternating point graph (feature refinement) and distribution graph (confidence refinement) for the configured number of layers. |
| `InitializeGraphParams` | Initializes dual graph parameters. |
| `MetaTrain(TaskBatch<,,>)` |  |
| `PropagateOneLayer(Vector<>,Vector<>,Int32)` | Performs one layer of graph propagation: computes pairwise similarity-based edge weights via softmax, then updates each node via weighted neighbor aggregation with a residual connection. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_distGraphParams` | Parameters for the distribution graph propagation layers. |
| `_pointGraphParams` | Parameters for the point graph propagation layers. |

