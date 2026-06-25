---
title: "Denclue<T>"
description: "Denclue<T> — Models & Types in AiDotNet.Clustering.Density."
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
    .ConfigureModel(new Denclue<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrix(data))
    .BuildAsync();

var labels = result.Predict(data);
Console.WriteLine($"Denclue: clustered {labels.Length} points.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Denclue(DenclueOptions<>)` | Initializes a new DENCLUE instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AttractorDensities` | Gets the density values at each attractor. |
| `Attractors` | Gets the density attractors (cluster centers). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `CreateNewInstance` |  |
| `DeepCopy` |  |
| `FitPredict(Matrix<>)` |  |
| `GetDensity(Vector<>)` | Gets the density estimate at a given point. |
| `GetOptions` |  |
| `Predict(Matrix<>)` |  |
| `Train(Matrix<>,Vector<>)` |  |
| `WithParameters(Vector<>)` |  |

