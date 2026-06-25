---
title: "IVFIndex<T>"
description: "Inverted File (IVF) index that partitions the vector space for faster search."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.VectorSearch.Indexes`

Inverted File (IVF) index that partitions the vector space for faster search.

## How It Works

IVF partitions vectors into clusters (cells) and only searches the most relevant
clusters during query time. This is an approximate nearest neighbor (ANN) method
that trades some accuracy for significant speed improvements.
Search complexity: O(n/m + k) where n is total vectors, m is number of clusters, k is result size.
Best for medium to large datasets (10K - 10M vectors).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `IVFIndex(ISimilarityMetric<>,Int32,Int32)` | Initializes a new instance of the IVFIndex class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Count` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Add(String,Vector<>)` |  |
| `AddBatch(Dictionary<String,Vector<>>)` |  |
| `Clear` |  |
| `Remove(String)` |  |
| `Search(Vector<>,Int32)` |  |

