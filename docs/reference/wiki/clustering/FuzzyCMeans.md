---
title: "FuzzyCMeans"
description: "Fuzzy C-Means (FCM) soft clustering implementation."
section: "Reference"
---

_Clustering_

Fuzzy C-Means (FCM) soft clustering implementation.

## For Beginners

FCM creates "soft" or "fuzzy" clusters. Instead of saying "Point X belongs to Cluster 1", FCM says: "Point X is 60% Cluster 1, 30% Cluster 2, 10% Cluster 3" This captures uncertainty and allows for overlapping clusters. The fuzziness parameter controls how much overlap is allowed. Use FCM when: - Clusters have unclear boundaries - Points naturally fit multiple categories - You need confidence/probability information

## How It Works

Fuzzy C-Means assigns each point a membership degree to each cluster, rather than a hard assignment. The memberships sum to 1 for each point. 

Algorithm: 1. Initialize membership matrix randomly 2. Compute cluster centers from weighted memberships 3. Update memberships based on distances to centers 4. Repeat until convergence

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
    .ConfigureModel(new FuzzyCMeans<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrix(data))
    .BuildAsync();

var labels = result.Predict(data);
Console.WriteLine($"FuzzyCMeans: clustered {labels.Length} points.");
```

