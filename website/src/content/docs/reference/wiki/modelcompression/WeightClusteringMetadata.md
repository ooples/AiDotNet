---
title: "WeightClusteringMetadata<T>"
description: "Metadata for weight clustering compression."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ModelCompression`

Metadata for weight clustering compression.

## For Beginners

This metadata stores the information needed to decompress clustered weights:

- The cluster centers (the actual weight values each cluster represents)
- The number of clusters used
- The original number of weights

When decompressing, each cluster index is replaced with its corresponding cluster center value.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `WeightClusteringMetadata([],Int32,Int32)` | Initializes a new instance of the WeightClusteringMetadata class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ClusterCenters` | Gets the cluster centers. |
| `NumClusters` | Gets the number of clusters. |
| `OriginalLength` | Gets the original length of the weights array. |
| `Type` | Gets the compression type. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetMetadataSize` | Gets the size in bytes of this metadata structure. |

