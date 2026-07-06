---
title: "Clustering"
description: "Group similar data points with AiDotNet."
order: 3
section: "Tutorials"
---


Learn to group similar data points with AiDotNet's clustering algorithms — all through the same `AiModelBuilder` facade.

## What is Clustering?

Clustering is an unsupervised learning task: group similar points together without predefined labels. Examples include customer segmentation, document organization, anomaly detection, and image segmentation.

Clustering uses the features-only loader `DataLoaders.FromMatrix(...)`. Cluster assignments come from `result.Predict(...)`, and quality metrics are cached on `result.Evaluation.ClusteringMetrics`.

## Types of Clustering

- **Partitioning** (`KMeans`, `KMedoids`): non-overlapping clusters around centers.
- **Density-based** (`DBSCAN`, `MeanShift`): dense regions separated by sparse ones; finds outliers.
- **Hierarchical** (`AgglomerativeClustering`, `BIRCH`): a tree of clusters.

---

## Quick Start (K-Means)

```csharp
using AiDotNet;
using AiDotNet.Clustering.Options;
using AiDotNet.Clustering.Partitioning;
using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.LinearAlgebra;

// Customer purchase patterns: [age, annual spend]
double[][] customers =
{
    new[] { 25.0, 45000.0 }, new[] { 35.0, 67000.0 }, new[] { 45.0, 89000.0 },
    new[] { 32.0, 52000.0 }, new[] { 28.0, 43000.0 }, new[] { 48.0, 92000.0 },
};
var data = ToMatrix(customers);

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new KMeans<double>(new KMeansOptions<double> { NumClusters = 3, Seed = 42 }))
    .ConfigureDataLoader(DataLoaders.FromMatrix(data))
    .BuildAsync();

var labels = result.Predict(data);
Console.WriteLine($"Customer 0 -> cluster {(int)labels[0]}");

var metrics = result.Evaluation.ClusteringMetrics;
if (metrics is not null)
    Console.WriteLine($"Silhouette: {metrics.Silhouette:F4}");

static Matrix<double> ToMatrix(double[][] rows)
{
    var m = new Matrix<double>(rows.Length, rows[0].Length);
    for (int i = 0; i < rows.Length; i++)
        for (int j = 0; j < rows[0].Length; j++)
            m[i, j] = rows[i][j];
    return m;
}
```

---

## Available Clustering Algorithms

| Family | Algorithms |
|:-------|:-----------|
| Partitioning | `KMeans`, `KMedoids`, `BisectingKMeans`, `FuzzyCMeans` |
| Density-based | `DBSCAN`, `MeanShift`, `Denclue` |
| Hierarchical | `AgglomerativeClustering`, `BIRCH`, `CURE` |
| Graph / other | `SpectralClustering`, `AffinityPropagation` |

---

## Finding the Optimal K

There is no magic `K` — sweep candidate values and keep the best silhouette, which the facade computes for you.

```csharp
using AiDotNet;
using AiDotNet.Clustering.Options;
using AiDotNet.Clustering.Partitioning;
using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.LinearAlgebra;

double[][] points =
{
    new[] { 1.0, 2.0 }, new[] { 1.5, 1.8 }, new[] { 1.0, 0.6 },
    new[] { 5.0, 8.0 }, new[] { 8.0, 8.0 }, new[] { 9.0, 11.0 },
    new[] { 8.0, 2.0 }, new[] { 10.0, 2.0 }, new[] { 9.0, 3.0 },
};
var data = ToMatrix(points);

int bestK = 2;
double bestSilhouette = double.NegativeInfinity;
for (int k = 2; k <= 5; k++)
{
    var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
        .ConfigureModel(new KMeans<double>(new KMeansOptions<double> { NumClusters = k, Seed = 42 }))
        .ConfigureDataLoader(DataLoaders.FromMatrix(data))
        .BuildAsync();

    double silhouette = result.Evaluation.ClusteringMetrics?.Silhouette ?? double.NaN;
    Console.WriteLine($"K={k}: Silhouette={silhouette:F4}");
    if (silhouette > bestSilhouette) { bestSilhouette = silhouette; bestK = k; }
}
Console.WriteLine($"Best K: {bestK}");

static Matrix<double> ToMatrix(double[][] rows)
{
    var m = new Matrix<double>(rows.Length, rows[0].Length);
    for (int i = 0; i < rows.Length; i++)
        for (int j = 0; j < rows[0].Length; j++)
            m[i, j] = rows[i][j];
    return m;
}
```

---

## DBSCAN — Density-Based Clustering

Finds clusters of arbitrary shape and flags outliers as `DBSCAN<double>.NoiseLabel` (`-1`).

```csharp
using AiDotNet;
using AiDotNet.Clustering.Density;
using AiDotNet.Clustering.Options;
using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.LinearAlgebra;

double[][] points =
{
    new[] { 1.0, 2.0 }, new[] { 1.5, 1.8 }, new[] { 1.2, 2.2 }, new[] { 1.1, 1.9 },
    new[] { 8.0, 8.0 }, new[] { 8.2, 7.8 }, new[] { 7.9, 8.1 }, new[] { 8.1, 8.3 },
    new[] { 25.0, 80.0 },
};
var data = ToMatrix(points);

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new DBSCAN<double>(new DBSCANOptions<double> { Epsilon = 1.0, MinPoints = 3 }))
    .ConfigureDataLoader(DataLoaders.FromMatrix(data))
    .BuildAsync();

var labels = result.Predict(data);
int outliers = 0;
for (int i = 0; i < labels.Length; i++)
    if ((int)labels[i] == DBSCAN<double>.NoiseLabel) outliers++;
Console.WriteLine($"Outliers: {outliers}");

static Matrix<double> ToMatrix(double[][] rows)
{
    var m = new Matrix<double>(rows.Length, rows[0].Length);
    for (int i = 0; i < rows.Length; i++)
        for (int j = 0; j < rows[0].Length; j++)
            m[i, j] = rows[i][j];
    return m;
}
```

---

## Hierarchical Clustering

```csharp
using AiDotNet;
using AiDotNet.Clustering.Hierarchical;
using AiDotNet.Clustering.Options;
using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.LinearAlgebra;

double[][] points =
{
    new[] { 1.0, 2.0 }, new[] { 1.5, 1.8 }, new[] { 1.0, 0.6 },
    new[] { 5.0, 8.0 }, new[] { 8.0, 8.0 }, new[] { 9.0, 11.0 },
};
var data = ToMatrix(points);

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new AgglomerativeClustering<double>(new HierarchicalOptions<double>
    {
        NumClusters = 3,
        Linkage = LinkageMethod.Ward
    }))
    .ConfigureDataLoader(DataLoaders.FromMatrix(data))
    .BuildAsync();

Console.WriteLine($"Davies-Bouldin: {result.Evaluation.ClusteringMetrics?.DaviesBouldin:F4}");

static Matrix<double> ToMatrix(double[][] rows)
{
    var m = new Matrix<double>(rows.Length, rows[0].Length);
    for (int i = 0; i < rows.Length; i++)
        for (int j = 0; j < rows[0].Length; j++)
            m[i, j] = rows[i][j];
    return m;
}
```

### Linkage Methods

| Method | Best For |
|:-------|:---------|
| `Ward` | Compact, equal-sized clusters |
| `Complete` | Well-separated clusters |
| `Average` | Balanced approach |
| `Single` | Elongated clusters |

---

## Evaluation Metrics

Every clustering build fills `result.Evaluation.ClusteringMetrics` with the standard internal validity measures.

```csharp
using AiDotNet;
using AiDotNet.Clustering.Options;
using AiDotNet.Clustering.Partitioning;
using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.LinearAlgebra;

double[][] points =
{
    new[] { 1.0, 2.0 }, new[] { 1.5, 1.8 }, new[] { 1.0, 0.6 },
    new[] { 5.0, 8.0 }, new[] { 8.0, 8.0 }, new[] { 9.0, 11.0 },
    new[] { 8.0, 2.0 }, new[] { 10.0, 2.0 }, new[] { 9.0, 3.0 },
};

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new KMeans<double>(new KMeansOptions<double> { NumClusters = 3, Seed = 42 }))
    .ConfigureDataLoader(DataLoaders.FromMatrix(ToMatrix(points)))
    .BuildAsync();

var m = result.Evaluation.ClusteringMetrics;
if (m is not null)
{
    Console.WriteLine($"Silhouette:        {m.Silhouette:F4}  (higher is better)");
    Console.WriteLine($"Calinski-Harabasz: {m.CalinskiHarabasz:F2}  (higher is better)");
    Console.WriteLine($"Davies-Bouldin:    {m.DaviesBouldin:F4}  (lower is better)");
}

static Matrix<double> ToMatrix(double[][] rows)
{
    var mtx = new Matrix<double>(rows.Length, rows[0].Length);
    for (int i = 0; i < rows.Length; i++)
        for (int j = 0; j < rows[0].Length; j++)
            mtx[i, j] = rows[i][j];
    return mtx;
}
```

---

## Best Practices

1. **Scale features**: clustering is distance-based; normalize before building (`ConfigurePreprocessing(...)`).
2. **Try multiple algorithms**: K-Means, DBSCAN, and hierarchical find different shapes.
3. **Validate K**: sweep and compare silhouette across candidates.
4. **Handle outliers**: prefer DBSCAN for noisy data.
5. **Interpret results**: inspect cluster sizes and centers.

---

## Next Steps

- [Customer Segmentation Example](/docs/examples/clustering/)
- [Classification Tutorial](/docs/tutorials/classification/) — for labeled data
