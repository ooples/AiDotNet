---
title: "LeidenResult"
description: "Contains the results of Leiden community detection, including hierarchical partitions."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.Graph.Communities`

Contains the results of Leiden community detection, including hierarchical partitions.

## For Beginners

The Leiden algorithm produces a hierarchy of communities:

- Level 0: Finest partition — each node belongs to a small community
- Level 1: Coarser — small communities merged into bigger ones
- Level N: Coarsest — entire graph in a few large communities

The Communities dictionary gives the final (finest) partition:
mapping each node ID to its community ID.

## Properties

| Property | Summary |
|:-----|:--------|
| `Communities` | Final community assignments: nodeId → communityId (finest level). |
| `HierarchicalPartitions` | Hierarchical partitions: level → (nodeId → communityId). |
| `ModularityScores` | Modularity score at each level of the hierarchy. |

