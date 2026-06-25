---
title: "ExpandedSubgraph<T>"
description: "Represents a subgraph expanded with pseudo-nodes."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Graph`

Represents a subgraph expanded with pseudo-nodes.

## Properties

| Property | Summary |
|:-----|:--------|
| `Adjacency` | Expanded adjacency matrix (flattened). |
| `NodeFeatures` | Expanded node features (flattened, [totalNodes * featureDim]). |
| `OriginalNodeCount` | Number of original nodes. |
| `PseudoNodeCount` | Number of pseudo-nodes added. |

