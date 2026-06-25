---
title: "LeidenCommunityDetector<T>"
description: "Implements the Leiden algorithm for community detection in knowledge graphs."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.Graph.Communities`

Implements the Leiden algorithm for community detection in knowledge graphs.

## For Beginners

Community detection finds groups of nodes that are more connected
to each other than to the rest of the graph.

Think of a school: students naturally form groups (sports team, drama club, study group).
The Leiden algorithm automatically discovers these groups by looking at who connects to whom.

It produces a hierarchy:

- Fine level: Small friend groups
- Coarser level: Clubs and teams
- Coarsest level: Grade levels or departments

## How It Works

The Leiden algorithm (Traag et al., 2019) improves on the Louvain algorithm by guaranteeing
well-connected communities. It consists of three phases repeated iteratively:

1. Local moving: greedily move nodes to maximize modularity
2. Refinement: ensure communities are internally connected
3. Aggregation: merge communities into super-nodes and repeat

## Methods

| Method | Summary |
|:-----|:--------|
| `Detect(KnowledgeGraph<>,LeidenOptions)` | Runs the Leiden algorithm on the given knowledge graph. |

