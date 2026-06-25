---
title: "DBSCAN<T>"
description: "DBSCAN<T> — Models & Types in AiDotNet.Clustering.Density."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Clustering.Density`

_No summary documentation available yet._

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
    .ConfigureModel(new DBSCAN<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrix(data))
    .BuildAsync();

var labels = result.Predict(data);
Console.WriteLine($"DBSCAN: clustered {labels.Length} points.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DBSCAN(DBSCANOptions<>)` | Initializes a new DBSCAN instance with the specified options. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CorePointMask` | Gets the mask indicating which points are core points. |
| `Epsilon` | Gets the epsilon neighborhood radius. |
| `MinPoints` | Gets the minimum points for core point classification. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `CreateNewInstance` |  |
| `DeepCopy` |  |
| `FitPredict(Matrix<>)` |  |
| `GetCoreSampleIndices` | Gets the indices of core samples. |
| `GetNoiseCount` | Gets the number of noise points (outliers). |
| `GetOptions` |  |
| `Predict(Matrix<>)` |  |
| `Train(Matrix<>,Vector<>)` |  |
| `Transform(Matrix<>)` |  |
| `WithParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `NoiseLabel` | Noise label (points not assigned to any cluster). |
| `UndefinedLabel` | Undefined label (not yet processed). |

