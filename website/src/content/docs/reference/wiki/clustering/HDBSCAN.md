---
title: "HDBSCAN<T>"
description: "HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Clustering.Density`

HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise).

## For Beginners

HDBSCAN automatically finds the right density level.

The key insight:

- Instead of picking ONE epsilon (like DBSCAN)
- Look at how clusters form at ALL epsilon values
- Pick clusters that "live longest" in the hierarchy

Core distance: How far to reach k nearest neighbors
Mutual reachability: Max of core distances and actual distance
This makes sparse points "push away" from dense clusters.

Benefits:

- No epsilon to tune
- Finds varying-density clusters
- Robust noise detection
- Provides cluster hierarchy

## How It Works

HDBSCAN extends DBSCAN by finding clusters at all density levels and selecting
the most stable clusters. It uses mutual reachability distance and minimum
spanning tree construction.

Algorithm steps:

1. Compute core distances (distance to k-th nearest neighbor)
2. Build mutual reachability graph
3. Construct minimum spanning tree
4. Build cluster hierarchy (condensed tree)
5. Extract clusters using stability

## Example

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Clustering.Density;
using AiDotNet.Tensors.LinearAlgebra;

var data = new Matrix<double>(6, 2);
double[][] rows = { new[] { 1.0, 1.0 }, new[] { 1.2, 0.9 }, new[] { 1.1, 1.1 },
                    new[] { 8.0, 8.0 }, new[] { 8.2, 7.9 }, new[] { 7.9, 8.1 } };
for (int i = 0; i < 6; i++) { data[i, 0] = rows[i][0]; data[i, 1] = rows[i][1]; }

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new HDBSCAN<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrix(data))
    .BuildAsync();

var labels = result.Predict(data);
Console.WriteLine($"HDBSCAN: clustered {labels.Length} points.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HDBSCAN(HDBSCANOptions<>)` | Initializes a new HDBSCAN instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `OutlierScores` | Gets the outlier scores for each point. |
| `Probabilities` | Gets the cluster membership probabilities. |
| `SupportsParameterInitialization` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `CreateNewInstance` |  |
| `DeepCopy` |  |
| `FitPredict(Matrix<>)` |  |
| `GetOptions` |  |
| `MarkDescendantsNotCluster(Int32,Dictionary<Int32,List<Int32>>,HashSet<Int32>,Dictionary<Int32,Boolean>)` | BFS deselect all descendants of a cluster per scikit-learn reference. |
| `Predict(Matrix<>)` |  |
| `Train(Matrix<>,Vector<>)` |  |
| `WithParameters(Vector<>)` |  |

