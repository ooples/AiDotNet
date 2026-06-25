---
title: "GraphAnalytics"
description: "Provides graph analytics algorithms for analyzing knowledge graphs."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.RetrievalAugmentedGeneration.Graph`

Provides graph analytics algorithms for analyzing knowledge graphs.

## For Beginners

Graph analytics help you understand your graph.

Think of a social network:

- PageRank: Who are the most influential people?
- Degree Centrality: Who has the most connections?
- Closeness Centrality: Who can reach everyone quickly?
- Betweenness Centrality: Who connects different groups?

These algorithms answer "who's important?" and "how are things connected?"

## How It Works

This class implements common graph algorithms used to analyze the structure
and importance of nodes and edges in a knowledge graph.

## Methods

| Method | Summary |
|:-----|:--------|
| `BreadthFirstSearchDistances(KnowledgeGraph<>,String)` | Performs breadth-first search to calculate distances from a source node to all others. |
| `CalculateAverageClusteringCoefficient(KnowledgeGraph<>)` | Calculates the average clustering coefficient for the entire graph. |
| `CalculateBetweennessCentrality(KnowledgeGraph<>,Boolean)` | Calculates betweenness centrality for all nodes in the graph. |
| `CalculateClosenessCentrality(KnowledgeGraph<>)` | Calculates closeness centrality for all nodes in the graph. |
| `CalculateClusteringCoefficient(KnowledgeGraph<>)` | Calculates the clustering coefficient for each node. |
| `CalculateDegreeCentrality(KnowledgeGraph<>,Boolean)` | Calculates degree centrality for all nodes in the graph. |
| `CalculatePageRank(KnowledgeGraph<>,Double,Int32,Double)` | Calculates PageRank scores for all nodes in the graph. |
| `DetectCommunitiesLabelPropagation(KnowledgeGraph<>,Int32)` | Detects communities using Label Propagation algorithm. |
| `FindConnectedComponents(KnowledgeGraph<>)` | Finds all connected components in the graph. |
| `GetTopKNodes(Dictionary<String,Double>,Int32)` | Identifies the top-k most central nodes based on a centrality measure. |

