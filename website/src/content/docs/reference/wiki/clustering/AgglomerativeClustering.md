---
title: "AgglomerativeClustering"
description: "Agglomerative Hierarchical Clustering implementation."
section: "Reference"
---

_Clustering_

Agglomerative Hierarchical Clustering implementation.

## For Beginners

Hierarchical clustering builds a "family tree" of your data. Imagine sorting photos of animals: 1. Start with each photo as its own group 2. Find the two most similar photos and group them 3. Keep grouping until you have the number of groups you want The result can be shown as a dendrogram (tree diagram) where: - Bottom: Each individual item - Top: All items merged into one group - Height: Shows how different merged groups are Use Ward linkage for most cases - it creates nice, balanced clusters.

## How It Works

Agglomerative clustering builds a hierarchy of clusters bottom-up. Starting with each sample as its own cluster, it iteratively merges the closest pair of clusters until the desired number of clusters is reached. 

Time complexity: O(n³) for naive implementation, O(n² log n) with efficient data structures. Space complexity: O(n²) for distance matrix.

## Example

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Clustering.Hierarchical;
using AiDotNet.Tensors.LinearAlgebra;

var data = new Matrix<double>(6, 2);
double[][] rows = { new[] { 1.0, 1.0 }, new[] { 1.2, 0.9 }, new[] { 1.1, 1.1 },
                    new[] { 8.0, 8.0 }, new[] { 8.2, 7.9 }, new[] { 7.9, 8.1 } };
for (int i = 0; i < 6; i++) { data[i, 0] = rows[i][0]; data[i, 1] = rows[i][1]; }

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new AgglomerativeClustering<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrix(data))
    .BuildAsync();

var labels = result.Predict(data);
Console.WriteLine($"AgglomerativeClustering: clustered {labels.Length} points.");
```

