---
title: "GraphNeighborhoodPrivacy<T>"
description: "Applies local differential privacy (LDP) to neighborhood queries to prevent topology leakage."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Graph`

Applies local differential privacy (LDP) to neighborhood queries to prevent topology leakage.

## For Beginners

GNN embeddings encode information about a node's neighborhood structure.
If shared naively, an adversary could reconstruct the local graph topology. This class adds
calibrated noise to neighborhood aggregations before they leave a client, ensuring that
individual edges cannot be inferred from the shared embeddings.

## How It Works

**Mechanism:**

**Privacy guarantee:** (epsilon, delta)-differential privacy for individual edges.
Lower epsilon = stronger privacy but more noise.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GraphNeighborhoodPrivacy(Double,Double,Double)` | Initializes a new instance of `GraphNeighborhoodPrivacy`. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ClipAndNoiseNeighborhood(Tensor<>,Double)` | Clips and adds noise to an aggregated neighborhood feature vector. |
| `PerturbDegrees(Int32[])` | Perturbs node degree information with Laplace noise. |
| `PrivatizeEmbeddings(Tensor<>,Int32)` | Adds calibrated Gaussian noise to node embeddings to protect neighborhood structure. |
| `RandomizedResponseEdge(Boolean)` | Applies randomized response to edge existence queries. |

