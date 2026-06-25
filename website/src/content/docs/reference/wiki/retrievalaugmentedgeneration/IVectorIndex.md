---
title: "IVectorIndex<T>"
description: "Interface for vector search indexes."
section: "API Reference"
---

`Interfaces` · `AiDotNet.RetrievalAugmentedGeneration.VectorSearch.Indexes`

Interface for vector search indexes.

## Properties

| Property | Summary |
|:-----|:--------|
| `Count` | Gets the number of vectors in the index. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Add(String,Vector<>)` | Adds a vector to the index. |
| `AddBatch(Dictionary<String,Vector<>>)` | Adds multiple vectors to the index in batch. |
| `Clear` | Clears all vectors from the index. |
| `Remove(String)` | Removes a vector from the index. |
| `Search(Vector<>,Int32)` | Searches for the k nearest neighbors to the query vector. |

