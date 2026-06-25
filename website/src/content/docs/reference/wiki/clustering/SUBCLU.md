---
title: "SUBCLU<T>"
description: "SUBCLU (SUBspace CLUstering) density-connected subspace clustering algorithm."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Clustering.Subspace`

SUBCLU (SUBspace CLUstering) density-connected subspace clustering algorithm.

## For Beginners

SUBCLU efficiently finds density-based clusters:

The key insight is "downward closure":

- If points are NOT a cluster in 2D, they can't be a cluster in 3D or higher
- So we can skip many subspace combinations!

Example:

- Check features 1 and 2: no cluster found
- Skip checking features 1, 2, and 3 (waste of time)
- Only check subspaces that might have clusters

This makes SUBCLU much faster than brute-force subspace search.

## How It Works

SUBCLU is a subspace clustering algorithm based on the DBSCAN density concept.
It exploits the monotonicity property: if a cluster exists in a k-dimensional
subspace, it must exist in all (k-1)-dimensional projections of that subspace.

Algorithm steps:

1. Apply DBSCAN to each 1-D subspace
2. Generate candidate 2-D subspaces from 1-D clusters
3. Apply DBSCAN to candidate subspaces
4. Use monotonicity to prune: no cluster in 2-D means skip higher dimensions
5. Continue until no more candidates or max dimension reached

## Example

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Clustering.Subspace;
using AiDotNet.Tensors.LinearAlgebra;

var data = new Matrix<double>(6, 2);
double[][] rows = { new[] { 1.0, 1.0 }, new[] { 1.2, 0.9 }, new[] { 1.1, 1.1 },
                    new[] { 8.0, 8.0 }, new[] { 8.2, 7.9 }, new[] { 7.9, 8.1 } };
for (int i = 0; i < 6; i++) { data[i, 0] = rows[i][0]; data[i, 1] = rows[i][1]; }

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new SUBCLU<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrix(data))
    .BuildAsync();

var labels = result.Predict(data);
Console.WriteLine($"SUBCLU: clustered {labels.Length} points.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SUBCLU(SUBCLUOptions<>)` | Initializes a new SUBCLU instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SubspaceClusters` | Gets the discovered subspace clusters. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `CreateNewInstance` |  |
| `DeepCopy` |  |
| `GetOptions` |  |
| `Predict(Matrix<>)` |  |
| `Train(Matrix<>,Vector<>)` |  |
| `WithParameters(Vector<>)` |  |

