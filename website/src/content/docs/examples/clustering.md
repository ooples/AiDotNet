---
title: "Clustering Example"
description: "Cluster data with K-Means, DBSCAN, and hierarchical clustering."
order: 3
section: "Examples"
---


This guide demonstrates how to use clustering algorithms for customer segmentation and pattern discovery using AiDotNet's `AiModelBuilder` facade.

## Overview

Clustering is unsupervised, so you have features but no labels. Every clustering model is built the same way as any other AiDotNet model — `ConfigureModel(...)` + `ConfigureDataLoader(...)` + `BuildAsync()` — except you use `DataLoaders.FromMatrix(...)`, the features-only loader. Cluster assignments come from `result.Predict(...)`, and quality metrics are computed for you and cached on `result.Evaluation.ClusteringMetrics`.

## Customer Segmentation (K-Means)

```csharp
using AiDotNet;
using AiDotNet.Clustering.Options;
using AiDotNet.Clustering.Partitioning;
using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.LinearAlgebra;

// Customer data: [Age, Annual Income ($K), Spending Score (1-100)]
double[][] customers =
{
    new[] { 19.0, 15.0, 39.0 }, new[] { 21.0, 15.0, 81.0 },
    new[] { 20.0, 16.0, 6.0 },  new[] { 23.0, 16.0, 77.0 },
    new[] { 31.0, 17.0, 40.0 }, new[] { 22.0, 17.0, 76.0 },
    new[] { 35.0, 18.0, 6.0 },  new[] { 23.0, 18.0, 94.0 },
    new[] { 64.0, 19.0, 3.0 },  new[] { 30.0, 19.0, 72.0 },
    new[] { 45.0, 20.0, 50.0 }, new[] { 50.0, 21.0, 15.0 },
};

var data = ToMatrix(customers);

// FromMatrix is the unsupervised (features-only) loader. ConfigureModel takes the
// clustering algorithm just like any other model.
var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new KMeans<double>(new KMeansOptions<double>
    {
        NumClusters = 5,
        Seed = 42
    }))
    .ConfigureDataLoader(DataLoaders.FromMatrix(data))
    .BuildAsync();

// One cluster index per customer.
var labels = result.Predict(data);

// Quality metrics are computed automatically and cached on result.Evaluation.
var metrics = result.Evaluation.ClusteringMetrics;
Console.WriteLine("Customer Segmentation Results:");
if (metrics is not null)
    Console.WriteLine($"  Silhouette Score: {metrics.Silhouette:F4} (range -1..1, higher is better)");

for (int i = 0; i < Math.Min(5, labels.Length); i++)
    Console.WriteLine($"  Customer {i}: cluster {(int)labels[i]}");

// Pack a jagged array into the dense Matrix the loader expects.
static Matrix<double> ToMatrix(double[][] rows)
{
    var m = new Matrix<double>(rows.Length, rows[0].Length);
    for (int i = 0; i < rows.Length; i++)
        for (int j = 0; j < rows[0].Length; j++)
            m[i, j] = rows[i][j];
    return m;
}
```

## Choosing the Number of Clusters

There is no `AutoDetectClusters` switch — instead, sweep `K` and keep the value with the best silhouette. Each candidate is a normal facade build, so the metric you compare on is the same `result.Evaluation.ClusteringMetrics.Silhouette` you would read in production.

```csharp
using AiDotNet;
using AiDotNet.Clustering.Options;
using AiDotNet.Clustering.Partitioning;
using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.LinearAlgebra;

double[][] points =
{
    new[] { 1.0, 2.0 },  new[] { 1.5, 1.8 }, new[] { 1.0, 0.6 },
    new[] { 5.0, 8.0 },  new[] { 8.0, 8.0 }, new[] { 9.0, 11.0 },
    new[] { 8.0, 2.0 },  new[] { 10.0, 2.0 }, new[] { 9.0, 3.0 },
};
var data = ToMatrix(points);

int bestK = 2;
double bestSilhouette = double.NegativeInfinity;

Console.WriteLine("Choosing K by silhouette:");
for (int k = 2; k <= 5; k++)
{
    var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
        .ConfigureModel(new KMeans<double>(new KMeansOptions<double> { NumClusters = k, Seed = 42 }))
        .ConfigureDataLoader(DataLoaders.FromMatrix(data))
        .BuildAsync();

    double silhouette = result.Evaluation.ClusteringMetrics?.Silhouette ?? double.NaN;
    Console.WriteLine($"  K={k}: Silhouette={silhouette:F4}");

    if (silhouette > bestSilhouette)
    {
        bestSilhouette = silhouette;
        bestK = k;
    }
}

Console.WriteLine($"Best K: {bestK} (silhouette {bestSilhouette:F4})");

static Matrix<double> ToMatrix(double[][] rows)
{
    var m = new Matrix<double>(rows.Length, rows[0].Length);
    for (int i = 0; i < rows.Length; i++)
        for (int j = 0; j < rows[0].Length; j++)
            m[i, j] = rows[i][j];
    return m;
}
```

## Density-Based Clustering (DBSCAN)

DBSCAN finds clusters of arbitrary shape and flags outliers — it labels noise points as `DBSCAN<double>.NoiseLabel` (`-1`), so you never have to guess a sentinel value.

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
    new[] { 25.0, 80.0 },  // an outlier far from both dense regions
};
var data = ToMatrix(points);

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new DBSCAN<double>(new DBSCANOptions<double>
    {
        Epsilon = 1.0,    // neighborhood radius
        MinPoints = 3     // minimum points to form a dense region
    }))
    .ConfigureDataLoader(DataLoaders.FromMatrix(data))
    .BuildAsync();

var labels = result.Predict(data);

int outliers = 0;
for (int i = 0; i < labels.Length; i++)
{
    if ((int)labels[i] == DBSCAN<double>.NoiseLabel)
    {
        outliers++;
        Console.WriteLine($"  Sample {i} is an outlier");
    }
}
Console.WriteLine($"Outliers detected: {outliers}");

static Matrix<double> ToMatrix(double[][] rows)
{
    var m = new Matrix<double>(rows.Length, rows[0].Length);
    for (int i = 0; i < rows.Length; i++)
        for (int j = 0; j < rows[0].Length; j++)
            m[i, j] = rows[i][j];
    return m;
}
```

## Hierarchical Clustering (Agglomerative)

Agglomerative clustering builds clusters bottom-up using a linkage rule. Configure it with `HierarchicalOptions` and pick a `LinkageMethod` (Ward, Complete, Average, …).

```csharp
using AiDotNet;
using AiDotNet.Clustering.Hierarchical;
using AiDotNet.Clustering.Options;
using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.LinearAlgebra;

double[][] points =
{
    new[] { 1.0, 2.0 },  new[] { 1.5, 1.8 }, new[] { 1.0, 0.6 },
    new[] { 5.0, 8.0 },  new[] { 8.0, 8.0 }, new[] { 9.0, 11.0 },
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

var labels = result.Predict(data);
var metrics = result.Evaluation.ClusteringMetrics;

Console.WriteLine("Hierarchical Clustering Results:");
if (metrics is not null)
    Console.WriteLine($"  Davies-Bouldin Index: {metrics.DaviesBouldin:F4} (lower is better)");

static Matrix<double> ToMatrix(double[][] rows)
{
    var m = new Matrix<double>(rows.Length, rows[0].Length);
    for (int i = 0; i < rows.Length; i++)
        for (int j = 0; j < rows[0].Length; j++)
            m[i, j] = rows[i][j];
    return m;
}
```

## Assigning New Data Points

Once trained, the model assigns new points to the nearest existing cluster through the same `result.Predict(...)` you use for any model.

```csharp
using AiDotNet;
using AiDotNet.Clustering.Options;
using AiDotNet.Clustering.Partitioning;
using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.LinearAlgebra;

double[][] trainingData =
{
    new[] { 19.0, 15.0, 39.0 }, new[] { 21.0, 15.0, 81.0 }, new[] { 20.0, 16.0, 6.0 },
    new[] { 23.0, 16.0, 77.0 }, new[] { 31.0, 17.0, 40.0 }, new[] { 22.0, 17.0, 76.0 },
};

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new KMeans<double>(new KMeansOptions<double> { NumClusters = 3, Seed = 42 }))
    .ConfigureDataLoader(DataLoaders.FromMatrix(ToMatrix(trainingData)))
    .BuildAsync();

// Assign new customers to existing clusters.
double[][] newCustomers =
{
    new[] { 25.0, 85.0, 90.0 },   // young, high income, high spending
    new[] { 55.0, 45.0, 20.0 },   // older, moderate income, low spending
};

var assignments = result.Predict(ToMatrix(newCustomers));
for (int i = 0; i < newCustomers.Length; i++)
    Console.WriteLine($"New customer {i + 1}: assigned to cluster {(int)assignments[i]}");

static Matrix<double> ToMatrix(double[][] rows)
{
    var m = new Matrix<double>(rows.Length, rows[0].Length);
    for (int i = 0; i < rows.Length; i++)
        for (int j = 0; j < rows[0].Length; j++)
            m[i, j] = rows[i][j];
    return m;
}
```

## Cluster Evaluation Metrics

Every clustering build fills in `result.Evaluation.ClusteringMetrics` with the standard internal validity measures — no separate evaluator to wire up.

```csharp
using AiDotNet;
using AiDotNet.Clustering.Options;
using AiDotNet.Clustering.Partitioning;
using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.LinearAlgebra;

double[][] points =
{
    new[] { 1.0, 2.0 },  new[] { 1.5, 1.8 }, new[] { 1.0, 0.6 },
    new[] { 5.0, 8.0 },  new[] { 8.0, 8.0 }, new[] { 9.0, 11.0 },
    new[] { 8.0, 2.0 },  new[] { 10.0, 2.0 }, new[] { 9.0, 3.0 },
};

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new KMeans<double>(new KMeansOptions<double> { NumClusters = 3, Seed = 42 }))
    .ConfigureDataLoader(DataLoaders.FromMatrix(ToMatrix(points)))
    .BuildAsync();

var metrics = result.Evaluation.ClusteringMetrics;
if (metrics is not null)
{
    Console.WriteLine("Cluster Quality Metrics:");
    Console.WriteLine($"  Silhouette Score:        {metrics.Silhouette:F4}  (range -1..1, higher is better)");
    Console.WriteLine($"  Calinski-Harabasz Index: {metrics.CalinskiHarabasz:F2}  (higher = better defined)");
    Console.WriteLine($"  Davies-Bouldin Index:    {metrics.DaviesBouldin:F4}  (lower = better separated)");
}

static Matrix<double> ToMatrix(double[][] rows)
{
    var m = new Matrix<double>(rows.Length, rows[0].Length);
    for (int i = 0; i < rows.Length; i++)
        for (int j = 0; j < rows[0].Length; j++)
            m[i, j] = rows[i][j];
    return m;
}
```

## Best Practices

1. **Normalize features**: clustering is sensitive to feature scales — add `ConfigurePreprocessing` or z-score your matrix before building.
2. **Use multiple metrics**: silhouette, Calinski-Harabasz, and Davies-Bouldin each tell part of the story.
3. **Validate with domain knowledge**: clusters should make business sense.
4. **Try multiple algorithms**: K-Means, DBSCAN, and hierarchical clustering reveal different structure.
5. **Handle outliers**: prefer DBSCAN when your data contains noise.

## Summary

AiDotNet exposes clustering through the same `AiModelBuilder` facade as everything else:

- Multiple algorithms — `KMeans`, `DBSCAN`, `AgglomerativeClustering`, and more — via `ConfigureModel(...)`
- The features-only `DataLoaders.FromMatrix(...)` loader for unsupervised data
- Cluster assignment through `result.Predict(...)`
- Quality metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin) auto-computed on `result.Evaluation.ClusteringMetrics`
- Outlier detection via `DBSCAN<double>.NoiseLabel`
