---
title: "KMeans<T>"
description: "K-Means clustering algorithm implementation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Clustering.Partitioning`

K-Means clustering algorithm implementation.

## For Beginners

K-Means finds k groups in your data by:

- Starting with k "center" points
- Grouping data points by their closest center
- Moving each center to the middle of its group
- Repeating until centers stop moving

It works best when:

- You know how many clusters to look for
- Clusters are roughly spherical and similar in size
- Data doesn't have many outliers

## How It Works

K-Means is one of the most widely used clustering algorithms. It partitions
n observations into k clusters by minimizing within-cluster variance (inertia).

Algorithm steps:

1. Initialize k cluster centers (randomly or using k-means++)
2. Assign each point to the nearest center
3. Update each center as the mean of its assigned points
4. Repeat steps 2-3 until convergence or max iterations

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
    .ConfigureModel(new KMeans<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrix(data))
    .BuildAsync();

var labels = result.Predict(data);
Console.WriteLine($"KMeans: clustered {labels.Length} points.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `KMeans(KMeansOptions<>)` | Initializes a new KMeans instance with the specified options. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumIterations` | Gets the number of iterations from the last fit. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |
| `FitPredict(Matrix<>)` |  |
| `GetOptions` |  |
| `Predict(Matrix<>)` |  |
| `Train(Matrix<>,Vector<>)` |  |
| `Transform(Matrix<>)` |  |
| `WithParameters(Vector<>)` |  |

