---
title: "GraphBasedFS<T>"
description: "Graph-based Feature Selection using feature relationships."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Graph`

Graph-based Feature Selection using feature relationships.

## For Beginners

Imagine features as people in a social network.
Features that are "well connected" (correlated with many other features)
might be key features. We use social network analysis techniques
to find the most "influential" features.

## How It Works

Constructs a graph where nodes are features and edges represent
relationships (like correlation). Uses graph centrality measures
to identify important features.

