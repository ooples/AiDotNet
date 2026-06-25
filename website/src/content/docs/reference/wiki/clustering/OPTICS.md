---
title: "OPTICS<T>"
description: "OPTICS<T> — Models & Types in AiDotNet.Clustering.Density."
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
    .ConfigureModel(new OPTICS<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrix(data))
    .BuildAsync();

var labels = result.Predict(data);
Console.WriteLine($"OPTICS: clustered {labels.Length} points.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OPTICS(OPTICSOptions<>)` | Initializes a new OPTICS instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CoreDistances` | Gets the core distances for each point. |
| `Ordering` | Gets the cluster ordering. |
| `ReachabilityDistances` | Gets the reachability distances in ordering. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `CreateNewInstance` |  |
| `DeepCopy` |  |
| `ExtractClustersAtEpsilon(Double)` | Extracts clusters at a different epsilon value. |
| `FitPredict(Matrix<>)` |  |
| `GetOptions` |  |
| `GetReachabilityPlot` | Gets the reachability plot data. |
| `Predict(Matrix<>)` |  |
| `WithParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `NoiseLabel` | Noise label for points not in any cluster. |

