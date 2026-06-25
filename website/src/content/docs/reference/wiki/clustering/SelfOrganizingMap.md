---
title: "SelfOrganizingMap<T>"
description: "SelfOrganizingMap<T> — Models & Types in AiDotNet.Clustering.Neural."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Clustering.Neural`

_No summary documentation available yet._

## Example

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Clustering.Neural;
using AiDotNet.Tensors.LinearAlgebra;

var data = new Matrix<double>(6, 2);
double[][] rows = { new[] { 1.0, 1.0 }, new[] { 1.2, 0.9 }, new[] { 1.1, 1.1 },
                    new[] { 8.0, 8.0 }, new[] { 8.2, 7.9 }, new[] { 7.9, 8.1 } };
for (int i = 0; i < 6; i++) { data[i, 0] = rows[i][0]; data[i, 1] = rows[i][1]; }

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new SelfOrganizingMap<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrix(data))
    .BuildAsync();

var labels = result.Predict(data);
Console.WriteLine($"SelfOrganizingMap: clustered {labels.Length} points.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SelfOrganizingMap(SOMOptions<>)` | Initializes a new SOM instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NeuronLabels` | Gets the cluster label assigned to each neuron (GridHeight x GridWidth). |
| `Weights` | Gets the neuron weight vectors (GridHeight x GridWidth x NumFeatures). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `CreateNewInstance` |  |
| `DeepCopy` |  |
| `FitPredict(Matrix<>)` |  |
| `GetGridPosition(Vector<>)` | Gets the 2D grid position for a data point. |
| `GetOptions` |  |
| `GetUMatrix` | Gets the U-Matrix (unified distance matrix) for visualization. |
| `Predict(Matrix<>)` |  |
| `Train(Matrix<>,Vector<>)` |  |
| `WithParameters(Vector<>)` |  |

