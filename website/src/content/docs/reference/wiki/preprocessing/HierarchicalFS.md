---
title: "HierarchicalFS<T>"
description: "Hierarchical Clustering-based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Clustering`

Hierarchical Clustering-based Feature Selection.

## For Beginners

This method builds a family tree of features,
starting with each feature as its own group, then gradually merging the
most similar ones. When we cut this tree at a certain height, we get
groups of related features. We pick the best feature from each group.

## How It Works

Uses hierarchical (agglomerative) clustering to build a tree of feature
relationships, then cuts the tree to create groups and selects
representative features from each group.

