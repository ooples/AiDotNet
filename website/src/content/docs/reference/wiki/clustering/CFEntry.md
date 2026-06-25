---
title: "CFEntry<T>"
description: "Represents a Clustering Feature (CF) entry."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Clustering.Hierarchical`

Represents a Clustering Feature (CF) entry.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CFEntry(Int32,[],)` | Initializes a new CFEntry. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Centroid` | Gets the centroid of this entry. |
| `LS` | Linear sum of points. |
| `N` | Number of points in the cluster. |
| `Radius` | Gets the radius of this entry. |
| `SS` | Sum of squared norms. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Distance(BIRCH<>.CFEntry,BIRCH<>.CFEntry)` | Computes distance between two CFEntries. |
| `FromPoint([])` | Creates a CFEntry from a single point. |
| `Merge(BIRCH<>.CFEntry,BIRCH<>.CFEntry)` | Merges two CFEntries. |

