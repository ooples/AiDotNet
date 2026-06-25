---
title: "AffinityPropagation<T>"
description: "AffinityPropagation<T> — Models & Types in AiDotNet.Clustering.Partitioning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Clustering.Partitioning`

_No summary documentation available yet._

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
    .ConfigureModel(new AffinityPropagation<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrix(data))
    .BuildAsync();

var labels = result.Predict(data);
Console.WriteLine($"AffinityPropagation: clustered {labels.Length} points.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AffinityPropagation(AffinityPropagationOptions<>)` | Initializes a new AffinityPropagation instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ExemplarIndices` | Gets the indices of exemplar points. |
| `SimilarityMatrix` | Gets the similarity matrix. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |
| `FitPredict(Matrix<>)` |  |
| `GetExemplars(Matrix<>)` | Gets the exemplar points as a matrix. |
| `GetOptions` |  |
| `Predict(Matrix<>)` |  |
| `Train(Matrix<>,Vector<>)` |  |
| `WithParameters(Vector<>)` |  |

