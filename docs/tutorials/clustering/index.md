---
layout: default
title: Clustering
parent: Tutorials
nav_order: 3
has_children: true
permalink: /tutorials/clustering/
---

# Clustering Tutorial
{: .no_toc }

Learn to group similar data points with AiDotNet's clustering algorithms.
{: .fs-6 .fw-300 }

---

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## What is Clustering?

Clustering is an unsupervised learning task where the goal is to group similar data points together without predefined labels. Examples include:
- Customer segmentation (grouping customers by behavior)
- Document organization (grouping similar documents)
- Anomaly detection (finding outliers)
- Image segmentation (grouping pixels)

## Types of Clustering

### Partitioning Methods
Divide data into non-overlapping clusters. Example: K-Means, K-Medoids

### Hierarchical Methods
Build a tree-like structure of clusters. Example: Agglomerative, BIRCH

### Density-Based Methods
Find clusters as dense regions separated by sparse regions. Example: DBSCAN, OPTICS

### Model-Based Methods
Assume data is generated from a mixture of distributions. Example: Gaussian Mixture Models

---

## Quick Start

```csharp
using AiDotNet;
using AiDotNet.Clustering;

// Prepare data - customer purchase patterns
var data = new double[][]
{
    new[] { 25.0, 45000.0 },  // Age, Annual spend
    new[] { 35.0, 67000.0 },
    new[] { 45.0, 89000.0 },
    new[] { 32.0, 52000.0 },
    new[] { 28.0, 43000.0 }
};

// Create K-Means clusterer
var kmeans = new KMeans<double>(new KMeansOptions<double>
{
    K = 3,
    MaxIterations = 100,
    Tolerance = 1e-4,
    InitMethod = KMeansInitMethod.KMeansPlusPlus
});

// Fit the model
kmeans.Fit(data);

// Get cluster assignments
var labels = kmeans.Labels;  // [0, 1, 2, 1, 0]

// Get cluster centers
var centers = kmeans.ClusterCenters;

// Predict cluster for new data
var newCustomer = new[] { 30.0, 55000.0 };
var cluster = kmeans.Predict(newCustomer);
Console.WriteLine($"Customer assigned to cluster: {cluster}");
```

---

## Available Clustering Algorithms

### Partitioning Methods

| Algorithm | Description | Best For |
|:----------|:------------|:---------|
| `KMeans` | Classic centroid-based clustering | Spherical clusters, large datasets |
| `KMedoids` | Uses actual data points as centers | Robust to outliers |
| `MiniBatchKMeans` | Scalable K-Means variant | Very large datasets |
| `FuzzyCMeans` | Soft clustering (membership degrees) | Overlapping clusters |

### Density-Based Methods

| Algorithm | Description | Best For |
|:----------|:------------|:---------|
| `DBSCAN` | Density-based spatial clustering | Arbitrary shapes, outlier detection |
| `HDBSCAN` | Hierarchical DBSCAN | Varying density clusters |
| `OPTICS` | Ordering points for cluster structure | Varying density, visualization |
| `MeanShift` | Mode-seeking algorithm | Unknown number of clusters |

### Hierarchical Methods

| Algorithm | Description | Best For |
|:----------|:------------|:---------|
| `AgglomerativeClustering` | Bottom-up hierarchical | Dendrogram visualization |
| `BIRCH` | Balanced iterative reducing | Large datasets, streaming |
| `BisectingKMeans` | Divisive hierarchical | Large datasets |

### Model-Based

| Algorithm | Description | Best For |
|:----------|:------------|:---------|
| `GaussianMixture` | Probabilistic clustering | Elliptical clusters, soft assignment |

---

## K-Means Clustering

The most popular clustering algorithm for its simplicity and speed.

### Basic Usage

```csharp
var kmeans = new KMeans<double>(new KMeansOptions<double>
{
    K = 3,
    InitMethod = KMeansInitMethod.KMeansPlusPlus,
    MaxIterations = 300,
    Tolerance = 1e-4,
    RandomState = 42  // For reproducibility
});

kmeans.Fit(data);
```

### Initialization Methods

```csharp
// K-Means++ (recommended)
InitMethod = KMeansInitMethod.KMeansPlusPlus

// Random initialization
InitMethod = KMeansInitMethod.Random

// Custom initial centroids
kmeans.Fit(data, initialCentroids: myCentroids);
```

### Finding Optimal K

```csharp
// Elbow method
var evaluator = new ClusteringEvaluator<double>();
var elbowResult = evaluator.ElbowMethod(data, kRange: Enumerable.Range(2, 10));
Console.WriteLine($"Optimal K: {elbowResult.OptimalK}");

// Silhouette analysis
var gapResult = evaluator.GapStatistic(data, kRange: Enumerable.Range(2, 10));
Console.WriteLine($"Optimal K (Gap): {gapResult.OptimalK}");
```

---

## DBSCAN - Density-Based Clustering

Excellent for clusters of arbitrary shape and automatic outlier detection.

```csharp
var dbscan = new DBSCAN<double>(new DBSCANOptions<double>
{
    Epsilon = 0.5,      // Maximum distance between neighbors
    MinPoints = 5       // Minimum points to form a cluster
});

dbscan.Fit(data);

// Labels: -1 indicates noise/outlier
var labels = dbscan.Labels;
var numClusters = labels.Where(l => l >= 0).Distinct().Count();
var numOutliers = labels.Count(l => l == -1);

Console.WriteLine($"Found {numClusters} clusters and {numOutliers} outliers");
```

### Choosing Epsilon

```csharp
// Use k-distance graph
var kDistances = dbscan.ComputeKDistances(data, k: 5);
// Plot and find the "elbow" - that's your epsilon
```

---

## Hierarchical Clustering

Build a hierarchy of clusters for deeper analysis.

```csharp
var agglom = new AgglomerativeClustering<double>(new HierarchicalOptions<double>
{
    NumClusters = 3,
    LinkageMethod = LinkageMethod.Ward,  // Minimize variance
    DistanceMetric = new EuclideanDistance<double>()
});

agglom.Fit(data);

// Get dendrogram for visualization
var dendrogram = agglom.GetDendrogram();
```

### Linkage Methods

| Method | Description | Best For |
|:-------|:------------|:---------|
| `Ward` | Minimizes variance increase | Compact, equal-sized clusters |
| `Complete` | Maximum pairwise distance | Well-separated clusters |
| `Average` | Mean pairwise distance | Balanced approach |
| `Single` | Minimum pairwise distance | Elongated clusters |

---

## Evaluation Metrics

### Internal Metrics (No Ground Truth)

```csharp
var evaluator = new ClusteringEvaluator<double>();

// Silhouette Score (-1 to 1, higher is better)
var silhouette = evaluator.SilhouetteScore(data, labels);
Console.WriteLine($"Silhouette Score: {silhouette:F4}");

// Calinski-Harabasz Index (higher is better)
var ch = evaluator.CalinskiHarabaszIndex(data, labels);
Console.WriteLine($"Calinski-Harabasz: {ch:F4}");

// Davies-Bouldin Index (lower is better)
var db = evaluator.DaviesBouldinIndex(data, labels);
Console.WriteLine($"Davies-Bouldin: {db:F4}");

// Within-cluster sum of squares
var wcss = evaluator.WCSS(data, labels, centers);
Console.WriteLine($"WCSS: {wcss:F4}");
```

### External Metrics (With Ground Truth)

```csharp
// Adjusted Rand Index (0 to 1, higher is better)
var ari = evaluator.AdjustedRandIndex(trueLabels, predictedLabels);

// Normalized Mutual Information (0 to 1, higher is better)
var nmi = evaluator.NormalizedMutualInformation(trueLabels, predictedLabels);

// Fowlkes-Mallows Index
var fmi = evaluator.FowlkesMallowsIndex(trueLabels, predictedLabels);
```

---

## Data Preprocessing for Clustering

### Feature Scaling (Critical!)

```csharp
// StandardScaler - zero mean, unit variance
var scaler = new StandardScaler<double>();
var scaledData = scaler.FitTransform(data);

// MinMaxScaler - scale to [0, 1]
var minMax = new MinMaxScaler<double>();
var normalizedData = minMax.FitTransform(data);
```

### Dimensionality Reduction

```csharp
// PCA before clustering
var pca = new PCA<double>(numComponents: 2);
var reducedData = pca.FitTransform(data);

// Then cluster
kmeans.Fit(reducedData);
```

---

## Distance Metrics

Choose the right distance metric for your data:

```csharp
// Euclidean (default) - continuous features
var euclidean = new EuclideanDistance<double>();

// Manhattan - robust to outliers
var manhattan = new ManhattanDistance<double>();

// Cosine - text/document clustering
var cosine = new CosineDistance<double>();

// Custom metric
var kmeans = new KMeans<double>(new KMeansOptions<double>
{
    K = 3,
    DistanceMetric = new CosineDistance<double>()
});
```

---

## Best Practices

1. **Always Scale Features**: Clustering is distance-based; features on different scales will dominate
2. **Try Multiple Algorithms**: Different algorithms find different cluster shapes
3. **Validate K Selection**: Use multiple methods (elbow, silhouette, gap statistic)
4. **Handle Outliers**: Consider DBSCAN or preprocessing to remove outliers
5. **Interpret Results**: Analyze cluster centers and characteristics

---

## Common Issues

### Clusters of Different Sizes

K-Means assumes equal-sized clusters. Use:
- DBSCAN/HDBSCAN for varying densities
- Gaussian Mixture Models with flexible covariance

### High-Dimensional Data

Curse of dimensionality affects distance calculations:
- Apply PCA/UMAP before clustering
- Use feature selection

### Choosing the Right K

No single best method exists:
```csharp
// Combine multiple approaches
var elbow = evaluator.ElbowMethod(data, 2..15);
var gap = evaluator.GapStatistic(data, 2..15);
var silhouettes = Enumerable.Range(2, 14)
    .Select(k => {
        var km = new KMeans<double>(new KMeansOptions<double> { K = k });
        km.Fit(data);
        return evaluator.SilhouetteScore(data, km.Labels);
    })
    .ToArray();

// Look for agreement across methods
```

---

## Next Steps

- [K-Means Sample](../../../samples/clustering/KMeans/)
- [DBSCAN Sample](../../../samples/clustering/DBSCAN/)
- [Customer Segmentation Example](../../../samples/clustering/CustomerSegmentation/)
- [Clustering API Reference](../../api/)
