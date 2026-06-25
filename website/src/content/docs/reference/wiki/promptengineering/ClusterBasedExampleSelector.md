---
title: "ClusterBasedExampleSelector<T>"
description: "Selects examples using a clustering approach to ensure broad coverage."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PromptEngineering.FewShot`

Selects examples using a clustering approach to ensure broad coverage.

## For Beginners

Groups similar examples and picks one from each group.

Think of organizing photos:

- Group 1: Beach photos
- Group 2: Mountain photos
- Group 3: City photos

Instead of showing all beach photos, you show one from each group.

Example:

Use this when:

- Examples naturally fall into categories
- You want guaranteed coverage of all categories
- Building a general-purpose system

## How It Works

This selector groups similar examples into clusters and selects representative examples from
each cluster. This ensures broad coverage across different types of examples.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ClusterBasedExampleSelector(Func<String,Vector<>>,Int32,Nullable<Int32>)` | Initializes a new instance of the ClusterBasedExampleSelector class. |
| `ClusterBasedExampleSelector(IEmbeddingModel<>,Int32,Nullable<Int32>)` | Initializes a new instance of the ClusterBasedExampleSelector class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ClusterCount` | Gets the number of clusters used for selection. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FindNearestCentroid(Vector<>,List<Vector<>>)` | Finds the nearest centroid for an embedding. |
| `GetClusterRelevance(List<FewShotExample>,Vector<>)` | Gets the average relevance of a cluster to the query. |
| `InitializeCentroids(List<Vector<>>,Int32,Int32)` | Initializes cluster centroids using k-means++ algorithm. |
| `OnExampleAdded(FewShotExample)` | Called when an example is added. |
| `OnExampleRemoved(FewShotExample)` | Called when an example is removed. |
| `RebuildClusters` | Rebuilds the clusters using k-means clustering. |
| `SelectExamplesCore(String,Int32)` | Selects examples from different clusters. |
| `UpdateCentroids(List<Vector<>>,Int32[],Int32,Int32)` | Updates centroids based on current assignments. |

