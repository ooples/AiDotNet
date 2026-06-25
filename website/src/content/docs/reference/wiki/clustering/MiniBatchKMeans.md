---
title: "MiniBatchKMeans<T>"
description: "Mini-Batch K-Means clustering algorithm for large-scale clustering."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Clustering.Partitioning`

Mini-Batch K-Means clustering algorithm for large-scale clustering.

## For Beginners

Think of Mini-Batch K-Means as "K-Means on a diet."

Instead of looking at all your data in each step (which is slow for big data),
it only looks at a random sample. This is much faster but gives nearly the
same result.

Speed comparison:

- 1 million points: Standard K-Means ~minutes, Mini-Batch ~seconds
- 10 million points: Standard K-Means ~hours, Mini-Batch ~minutes

The trade-off is slightly less optimal clustering, but usually the
difference is very small (a few percent in inertia).

## How It Works

Mini-Batch K-Means uses small random batches of data to update cluster centers,
making it much faster than standard K-Means for large datasets while producing
similar results.

Algorithm steps:

1. Initialize k cluster centers
2. For each iteration:

a. Sample a mini-batch of points
b. Assign each sample to its nearest center
c. Update centers using a gradient descent-like step

3. Repeat until convergence or max iterations

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
    .ConfigureModel(new MiniBatchKMeans<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrix(data))
    .BuildAsync();

var labels = result.Predict(data);
Console.WriteLine($"MiniBatchKMeans: clustered {labels.Length} points.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MiniBatchKMeans(MiniBatchKMeansOptions<>)` | Initializes a new MiniBatchKMeans instance with the specified options. |

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
| `PartialFit(Matrix<>)` | Performs a partial fit on a batch of data (for online/streaming learning). |
| `Predict(Matrix<>)` |  |
| `Train(Matrix<>,Vector<>)` |  |
| `Transform(Matrix<>)` |  |
| `WithParameters(Vector<>)` |  |

