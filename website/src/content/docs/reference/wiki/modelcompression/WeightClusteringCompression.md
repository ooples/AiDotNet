---
title: "WeightClusteringCompression<T>"
description: "Implements weight clustering compression using K-means clustering to group similar weights."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ModelCompression`

Implements weight clustering compression using K-means clustering to group similar weights.

## For Beginners

Weight clustering is like organizing a messy toolbox.

Imagine you have thousands of screws that are almost the same size:

- Some are 2.01mm, some 2.02mm, some 2.03mm, etc.
- Instead of keeping track of each exact size, you group similar sizes together
- You replace all sizes in a group with one representative size (like 2.0mm)

For neural networks:

- Instead of storing millions of slightly different weight values
- We group similar weights into clusters (like 256 or 512 groups)
- Each weight is replaced with its cluster center
- Instead of storing millions of unique values, we store which cluster each weight belongs to

This dramatically reduces storage because:

- Cluster IDs are much smaller than full weight values (8 bits vs 32 bits)
- We only need to store the cluster centers once

The result is a much smaller model that performs almost the same as the original!

## How It Works

Weight clustering reduces model size by identifying groups of similar weight values and replacing
them with their cluster representatives. This technique can achieve significant compression ratios
(10-50x) while maintaining model accuracy.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `WeightClusteringCompression(Int32,Int32,Double,Nullable<Int32>)` | Initializes a new instance of the WeightClusteringCompression class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateInertia(Vector<>,[],Int32[])` | Calculates the total inertia (sum of squared distances to cluster centers). |
| `Compress(Vector<>)` | Compresses weights using K-means clustering. |
| `Decompress(Vector<>,ICompressionMetadata<>)` | Decompresses weights by mapping cluster assignments back to cluster centers. |
| `FindNearestCluster(Double,[])` | Finds the nearest cluster center for a given weight value. |
| `GetCompressedSize(Vector<>,ICompressionMetadata<>)` | Gets the compressed size including cluster centers and assignments. |
| `InitializeClusterCentersKMeansPlusPlus(Vector<>,Int32)` | Initializes cluster centers using the K-means++ algorithm for better initial placement. |
| `PerformKMeansClustering(Vector<>,Int32)` | Performs K-means clustering on the weights. |

