---
title: "BIRCH"
description: "BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies) clustering."
section: "Reference"
---

_Clustering_

BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies) clustering.

## For Beginners

BIRCH creates a compressed summary of your data. The process: 1. Build a tree where each node summarizes nearby points 2. The tree automatically adjusts to fit memory constraints 3. Use the tree for fast approximate clustering CF (Clustering Feature) = (N, LS, SS) where: - N = count of points - LS = sum of all point coordinates - SS = sum of squared coordinates From these, you can compute: - Centroid = LS / N - Radius = sqrt((SS/N) - (LS/N)²)

## How It Works

BIRCH is designed for very large datasets. It builds a CF (Clustering Feature) tree that summarizes the data, then optionally applies a global clustering algorithm to the leaf entries. 

Each node in the CF-tree stores: - N: Number of points - LS: Linear Sum (vector sum of points) - SS: Squared Sum (sum of squared norms)

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
    .ConfigureModel(new BIRCH<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrix(data))
    .BuildAsync();

var labels = result.Predict(data);
Console.WriteLine($"BIRCH: clustered {labels.Length} points.");
```

