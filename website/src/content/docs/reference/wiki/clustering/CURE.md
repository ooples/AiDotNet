---
title: "CURE<T>"
description: "CURE (Clustering Using REpresentatives) hierarchical clustering algorithm."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Clustering.Hierarchical`

CURE (Clustering Using REpresentatives) hierarchical clustering algorithm.

## For Beginners

CURE finds clusters that aren't round:

Traditional clustering (like K-Means) assumes round clusters.
But real data often has:

- Banana-shaped clusters
- Spiral clusters
- Elongated clusters

CURE solves this by:

1. Using multiple "marker" points per cluster, not just one center
2. Placing these markers throughout the cluster
3. Measuring cluster similarity by comparing markers

This way, two banana-shaped clusters can be recognized as separate,
even if their centers are close together!

## How It Works

CURE is an agglomerative hierarchical clustering algorithm that uses multiple
representative points per cluster to better capture non-spherical cluster shapes.
Representatives are shrunk toward the cluster center to reduce sensitivity to outliers.

Algorithm steps:

1. Start with each point as its own cluster
2. Select representative points for each cluster (well-scattered)
3. Shrink representatives toward cluster center
4. Find and merge the two clusters with closest representatives
5. Repeat until desired number of clusters is reached

## Example

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Clustering.Hierarchical;
using AiDotNet.Tensors.LinearAlgebra;

var data = new Matrix<double>(6, 2);
double[][] rows = { new[] { 1.0, 1.0 }, new[] { 1.2, 0.9 }, new[] { 1.1, 1.1 },
                    new[] { 8.0, 8.0 }, new[] { 8.2, 7.9 }, new[] { 7.9, 8.1 } };
for (int i = 0; i < 6; i++) { data[i, 0] = rows[i][0]; data[i, 1] = rows[i][1]; }

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new CURE<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrix(data))
    .BuildAsync();

var labels = result.Predict(data);
Console.WriteLine($"CURE: clustered {labels.Length} points.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CURE(CUREOptions<>)` | Initializes a new CURE instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsParameterInitialization` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `CreateNewInstance` |  |
| `DeepCopy` |  |
| `GetOptions` |  |
| `Predict(Matrix<>)` |  |
| `RebuildClustersFromLabels(Matrix<>)` | Rebuilds the internal _clusters list from the current Labels after degenerate merge. |
| `Train(Matrix<>,Vector<>)` |  |
| `WithParameters(Vector<>)` |  |

