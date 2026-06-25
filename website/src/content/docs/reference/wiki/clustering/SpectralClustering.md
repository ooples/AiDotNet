---
title: "SpectralClustering<T>"
description: "Spectral Clustering implementation using graph Laplacian eigendecomposition."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Clustering.Spectral`

Spectral Clustering implementation using graph Laplacian eigendecomposition.

## For Beginners

Spectral clustering is like finding communities in a network.

The key insight: Similar points should be in the same cluster.
Instead of looking at distances directly, we:

1. Build a "friendship network" where similar points are connected
2. Find the natural groups in this network
3. Use these groups as clusters

This works better than K-Means when:

- Clusters aren't round/spherical
- Clusters have complex shapes (moons, spirals)
- You care about connectivity, not just distance

## How It Works

Spectral clustering works by:

1. Building an affinity (similarity) matrix
2. Computing the graph Laplacian
3. Finding eigenvectors corresponding to smallest eigenvalues
4. Clustering points in the reduced eigenspace

## Example

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Clustering.Spectral;
using AiDotNet.Tensors.LinearAlgebra;

var data = new Matrix<double>(6, 2);
double[][] rows = { new[] { 1.0, 1.0 }, new[] { 1.2, 0.9 }, new[] { 1.1, 1.1 },
                    new[] { 8.0, 8.0 }, new[] { 8.2, 7.9 }, new[] { 7.9, 8.1 } };
for (int i = 0; i < 6; i++) { data[i, 0] = rows[i][0]; data[i, 1] = rows[i][1]; }

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new SpectralClustering<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrix(data))
    .BuildAsync();

var labels = result.Predict(data);
Console.WriteLine($"SpectralClustering: clustered {labels.Length} points.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SpectralClustering(SpectralOptions<>)` | Initializes a new SpectralClustering instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AffinityMatrix` | Gets the affinity matrix. |
| `Embedding` | Gets the spectral embedding. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |
| `FitPredict(Matrix<>)` |  |
| `GetOptions` |  |
| `GetSpectralEmbedding` | Gets the spectral embedding of the data. |
| `Predict(Matrix<>)` |  |
| `Train(Matrix<>,Vector<>)` |  |
| `WithParameters(Vector<>)` |  |

