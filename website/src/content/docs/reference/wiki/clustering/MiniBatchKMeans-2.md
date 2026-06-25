---
title: "MiniBatchKMeans<T>"
description: "MiniBatchKMeans<T> — Models & Types in AiDotNet.Clustering.Streaming."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Clustering.Streaming`

_No summary documentation available yet._

## Example

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Clustering.Streaming;
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
| `MiniBatchKMeans(MiniBatchKMeansOptions<>)` | Initializes a new Mini-Batch K-Means instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FinalInertia` | Gets the final inertia (sum of squared distances to nearest center). |
| `IterationsPerformed` | Gets the number of iterations performed. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |
| `FitPredict(Matrix<>)` |  |
| `GetOptions` |  |
| `PartialFit(Matrix<>)` | Performs incremental training with additional data. |
| `Predict(Matrix<>)` |  |
| `Train(Matrix<>,Vector<>)` |  |
| `WithParameters(Vector<>)` |  |

