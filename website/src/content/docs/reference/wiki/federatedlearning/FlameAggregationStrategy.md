---
title: "FlameAggregationStrategy<T>"
description: "Implements FLAME (Filtering via cosine similarity + Adaptive clipping + Noise) for Byzantine-robust federated learning with backdoor resistance."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Aggregators`

Implements FLAME (Filtering via cosine similarity + Adaptive clipping + Noise) for
Byzantine-robust federated learning with backdoor resistance.

## For Beginners

Backdoor attacks in federated learning try to implant hidden
triggers in the global model. FLAME defends against this with a three-step approach:
(1) use HDBSCAN-inspired clustering on cosine distances to identify the honest majority
cluster, (2) clip surviving updates to a common norm to prevent magnitude-based attacks,
and (3) add calibrated noise to the aggregated result to erase any residual backdoor signal.

## How It Works

Pipeline:

Reference: Nguyen, T. D., et al. (2022). "FLAME: Taming Backdoors in Federated
Learning." USENIX Security 2022.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FlameAggregationStrategy(Double,Int32,Int32)` | Initializes a new instance of the `FlameAggregationStrategy` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MinClusterSize` | Gets the minimum cluster size for HDBSCAN. |
| `NoiseMultiplier` | Gets the noise multiplier for backdoor-erasure noise injection (not formal DP). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Aggregate(Dictionary<Int32,Dictionary<String,[]>>,Dictionary<Int32,Double>)` |  |
| `BuildMST(Double[0:,0:],Int32)` | Builds a minimum spanning tree using Prim's algorithm on the given distance matrix. |
| `GetStrategyName` |  |
| `IdentifyHonestCluster(Double[][],Double[],Int32,Int32)` | Identifies the honest majority cluster using HDBSCAN-inspired density-based clustering on cosine distances between client update vectors. |

