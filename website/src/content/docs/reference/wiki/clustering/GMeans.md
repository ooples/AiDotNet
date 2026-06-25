---
title: "GMeans<T>"
description: "GMeans<T> — Models & Types in AiDotNet.Clustering.AutoK."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Clustering.AutoK`

_No summary documentation available yet._

## Example

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Clustering.AutoK;
using AiDotNet.Tensors.LinearAlgebra;

var data = new Matrix<double>(6, 2);
double[][] rows = { new[] { 1.0, 1.0 }, new[] { 1.2, 0.9 }, new[] { 1.1, 1.1 },
                    new[] { 8.0, 8.0 }, new[] { 8.2, 7.9 }, new[] { 7.9, 8.1 } };
for (int i = 0; i < 6; i++) { data[i, 0] = rows[i][0]; data[i, 1] = rows[i][1]; }

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new GMeans<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrix(data))
    .BuildAsync();

var labels = result.Predict(data);
Console.WriteLine($"GMeans: clustered {labels.Length} points.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GMeans(GMeansOptions<>)` | Initializes a new GMeans instance. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |
| `FitPredict(Matrix<>)` |  |
| `GetOptions` |  |
| `Predict(Matrix<>)` |  |
| `Train(Matrix<>,Vector<>)` |  |
| `WithParameters(Vector<>)` |  |

