---
title: "LSHIndex<T>"
description: "Locality-Sensitive Hashing (LSH) index for approximate nearest neighbor search."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.VectorSearch.Indexes`

Locality-Sensitive Hashing (LSH) index for approximate nearest neighbor search.

## How It Works

LSH uses hash functions that map similar vectors to the same hash buckets.
Multiple hash tables are used to improve recall. During search, only vectors
in the same buckets as the query are considered, providing sublinear search time.
Search complexity: O(n^ρ) where ρ < 1, depending on parameters.
Best for high-dimensional sparse data and when approximate results are acceptable.
This is a simplified implementation using random projection for testing.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LSHIndex(ISimilarityMetric<>,Int32,Int32,Int32)` | Initializes a new instance of the LSHIndex class. |

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
| `RemoveFromHashTables(String,Vector<>)` | Removes an ID from all hash tables based on the given vector's hash. |
| `Search(Vector<>,Int32)` |  |

