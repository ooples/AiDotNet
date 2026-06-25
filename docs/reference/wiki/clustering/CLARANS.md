---
title: "CLARANS"
description: "CLARANS (Clustering Large Applications based on Randomized Search) implementation."
section: "Reference"
---

_Clustering_

CLARANS (Clustering Large Applications based on Randomized Search) implementation.

## For Beginners

CLARANS is like K-Means but uses actual data points.

Key differences from K-Means:

- Cluster centers (medoids) are real data points
- More robust: outliers don't pull centers as much
- Any distance metric works (not just Euclidean)

The "randomized" part:

- Instead of checking ALL possible swaps (slow!)
- Randomly sample potential swaps
- Still finds good solutions, just faster

Think of finding the best meeting spot for friends:

- Must be at someone's house (medoid = actual point)
- Try swapping whose house, keep improvements

## How It Works

CLARANS is a partitioning algorithm that uses medoids (actual data points)
as cluster centers. It improves on PAM by using randomized search to
explore the solution space more efficiently.

The algorithm:

1. Start with random medoids
2. For each iteration, consider swapping a medoid with a non-medoid
3. Accept swaps that reduce total cost
4. Repeat from different starting points, keep best result

## Example

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Clustering.Partitioning;
using AiDotNet.Tensors.LinearAlgebra;

var data = new Matrix<double>(6, 2);
double[][] rows = { new[] { 1.0, 1.0 }, new[] { 1.2, 0.9 }, new[] { 1.1, 1.1 },
                    new[] { 8.0, 8.0 }, new[] { 8.2, 7.9 }, new[] { 7.9, 8.1 } };
for (int i = 0; i < 6; i++) { data[i, 0] = rows[i][0]; data[i, 1] = rows[i][1]; }

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new CLARANS<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrix(data))
    .BuildAsync();

var labels = result.Predict(data);
Console.WriteLine($"CLARANS: clustered {labels.Length} points.");
```

