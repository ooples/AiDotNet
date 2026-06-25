---
title: "SeededKMeans<T>"
description: "Seeded K-Means implementation with labeled initialization points."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Clustering.SemiSupervised`

Seeded K-Means implementation with labeled initialization points.

## For Beginners

This is K-Means with a "head start."

Instead of guessing where clusters might be:

- You provide example points with known labels
- The algorithm learns where each cluster should be
- Then it clusters the rest of the data

Example: Classifying news articles

- Manually label 20 articles as "sports", "politics", "tech"
- Seeded K-Means learns what each category looks like
- Then automatically categorizes 10,000 more articles

Benefits:

- Better initialization = better results
- Incorporates domain knowledge
- Helps when random init gives poor results

## How It Works

Seeded K-Means uses pre-labeled data points to initialize cluster centers.
Instead of random initialization, it computes initial centers from known
cluster assignments, then proceeds with standard K-Means iterations.

Algorithm:

1. Group seed points by their labels
2. Compute initial centers as mean of each seed group
3. Run standard K-Means from these initial centers
4. Optionally constrain seed points to stay in original clusters

## Example

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Clustering.SemiSupervised;
using AiDotNet.Tensors.LinearAlgebra;

var data = new Matrix<double>(6, 2);
double[][] rows = { new[] { 1.0, 1.0 }, new[] { 1.2, 0.9 }, new[] { 1.1, 1.1 },
                    new[] { 8.0, 8.0 }, new[] { 8.2, 7.9 }, new[] { 7.9, 8.1 } };
for (int i = 0; i < 6; i++) { data[i, 0] = rows[i][0]; data[i, 1] = rows[i][1]; }

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new SeededKMeans<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrix(data))
    .BuildAsync();

var labels = result.Predict(data);
Console.WriteLine($"SeededKMeans: clustered {labels.Length} points.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SeededKMeans(SeededKMeansOptions<>)` | Initializes a new Seeded K-Means instance. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |
| `FitPredict(Matrix<>)` |  |
| `GetOptions` |  |
| `Predict(Matrix<>)` |  |
| `Train(Matrix<>,Vector<>)` |  |
| `WithParameters(Vector<>)` |  |

