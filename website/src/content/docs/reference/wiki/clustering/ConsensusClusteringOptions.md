---
title: "ConsensusClusteringOptions<T>"
description: "Configuration options for Consensus (Ensemble) Clustering."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Clustering.Ensemble`

Configuration options for Consensus (Ensemble) Clustering.

## For Beginners

Ensemble clustering is like taking a vote.

The idea:

1. Run multiple clustering algorithms (or same algorithm multiple times)
2. Each gives a different answer
3. Combine the answers to get a more reliable result

Why it works:

- Different algorithms have different strengths
- Random initialization can give different results
- Combining reduces the impact of any single bad result

Common approaches:

- Co-association matrix: How often are points clustered together?
- Voting: Which cluster assignment is most popular?

## How It Works

Consensus clustering combines multiple clustering solutions to produce
a more robust final clustering. It works by aggregating partitions from
different algorithms or the same algorithm with different parameters.

## Properties

| Property | Summary |
|:-----|:--------|
| `FinalAlgorithm` | Gets or sets the final clustering algorithm. |
| `Method` | Gets or sets the consensus method. |
| `NumBaseClusterings` | Gets or sets the number of base clusterings to generate. |
| `NumClusters` | Gets or sets the target number of clusters. |

