# Clustering with AiModelBuilder

This guide demonstrates how to use clustering algorithms for customer segmentation and pattern discovery using AiDotNet.

## Overview

AiDotNet provides clustering capabilities through the `AiModelBuilder` facade, making it easy to discover patterns in unlabeled data.

## Customer Segmentation

```csharp
using AiDotNet;

// Customer data: [Age, Annual Income ($K), Spending Score (1-100)]
var customers = new double[][]
{
    new[] { 19.0, 15.0, 39.0 },
    new[] { 21.0, 15.0, 81.0 },
    new[] { 20.0, 16.0, 6.0 },
    new[] { 23.0, 16.0, 77.0 },
    new[] { 31.0, 17.0, 40.0 },
    new[] { 22.0, 17.0, 76.0 },
    new[] { 35.0, 18.0, 6.0 },
    new[] { 23.0, 18.0, 94.0 },
    new[] { 64.0, 19.0, 3.0 },
    new[] { 30.0, 19.0, 72.0 },
    // ... more customers
};

// Build clustering model
var result = await new AiModelBuilder<double, double[][], int[]>()
    .ConfigureClustering(config =>
    {
        config.Algorithm = ClusteringAlgorithm.KMeans;
        config.NumClusters = 5;
    })
    .ConfigurePreprocessing(config =>
    {
        config.NormalizeFeatures = true;
    })
    .BuildAsync(customers);

// Get cluster assignments
var clusterLabels = result.ClusterLabels;

// View cluster statistics
Console.WriteLine("Customer Segmentation Results:");
Console.WriteLine($"Number of clusters: {result.NumClusters}");
Console.WriteLine($"Silhouette Score: {result.SilhouetteScore:F4}");

foreach (var cluster in result.ClusterSummaries)
{
    Console.WriteLine($"\nCluster {cluster.Id} ({cluster.Size} customers):");
    Console.WriteLine($"  Avg Age: {cluster.Centroid[0]:F1}");
    Console.WriteLine($"  Avg Income: ${cluster.Centroid[1]:F1}K");
    Console.WriteLine($"  Avg Spending: {cluster.Centroid[2]:F1}");
}
```

## Automatic Cluster Detection

```csharp
using AiDotNet;

// Sample data: your feature vectors
var data = new double[][]
{
    new[] { 1.0, 2.0 }, new[] { 1.5, 1.8 }, new[] { 5.0, 8.0 },
    new[] { 8.0, 8.0 }, new[] { 1.0, 0.6 }, new[] { 9.0, 11.0 }
};

// Let the algorithm find the optimal number of clusters
var result = await new AiModelBuilder<double, double[][], int[]>()
    .ConfigureClustering(config =>
    {
        config.Algorithm = ClusteringAlgorithm.KMeans;
        config.AutoDetectClusters = true;
        config.MinClusters = 2;
        config.MaxClusters = 10;
    })
    .ConfigurePreprocessing(config =>
    {
        config.NormalizeFeatures = true;
    })
    .BuildAsync(data);

Console.WriteLine($"Optimal number of clusters: {result.NumClusters}");
Console.WriteLine($"Selection method: {result.ClusterSelectionMethod}");

// View elbow analysis
Console.WriteLine("\nElbow Analysis:");
foreach (var point in result.ElbowAnalysis)
{
    Console.WriteLine($"K={point.K}: Inertia={point.Inertia:F2}, Silhouette={point.Silhouette:F4}");
}
```

## Density-Based Clustering (DBSCAN)

```csharp
using AiDotNet;
using System.Linq;

// Sample data: your feature vectors
var data = new double[][]
{
    new[] { 1.0, 2.0 }, new[] { 1.5, 1.8 }, new[] { 5.0, 8.0 },
    new[] { 8.0, 8.0 }, new[] { 1.0, 0.6 }, new[] { 9.0, 11.0 }
};

// DBSCAN for finding clusters of arbitrary shape
var result = await new AiModelBuilder<double, double[][], int[]>()
    .ConfigureClustering(config =>
    {
        config.Algorithm = ClusteringAlgorithm.DBSCAN;
        config.Epsilon = 0.5;       // Neighborhood radius
        config.MinSamples = 5;      // Minimum points to form cluster
    })
    .ConfigurePreprocessing(config =>
    {
        config.NormalizeFeatures = true;
    })
    .BuildAsync(data);

Console.WriteLine($"Clusters found: {result.NumClusters}");
Console.WriteLine($"Outliers detected: {result.NumOutliers}");

// Outliers are labeled as -1
var outlierIndices = result.ClusterLabels
    .Select((label, index) => new { label, index })
    .Where(x => x.label == -1)
    .Select(x => x.index)
    .ToArray();

Console.WriteLine($"\nOutlier samples: {string.Join(", ", outlierIndices)}");
```

## Hierarchical Clustering

```csharp
using AiDotNet;

// Hierarchical clustering with dendrogram
var result = await new AiModelBuilder<double, double[][], int[]>()
    .ConfigureClustering(config =>
    {
        config.Algorithm = ClusteringAlgorithm.Hierarchical;
        config.NumClusters = 4;
        config.Linkage = LinkageType.Ward;
    })
    .ConfigurePreprocessing(config =>
    {
        config.NormalizeFeatures = true;
    })
    .BuildAsync(data);

Console.WriteLine("Hierarchical Clustering Results:");
Console.WriteLine($"Number of clusters: {result.NumClusters}");

// View cluster hierarchy
Console.WriteLine("\nCluster Hierarchy:");
foreach (var node in result.DendrogramNodes)
{
    Console.WriteLine($"Merge: {node.Left} + {node.Right} at distance {node.Distance:F4}");
}
```

## Gaussian Mixture Models

```csharp
using AiDotNet;

// GMM for soft clustering (probability of belonging to each cluster)
var result = await new AiModelBuilder<double, double[][], double[][]>()
    .ConfigureClustering(config =>
    {
        config.Algorithm = ClusteringAlgorithm.GaussianMixture;
        config.NumClusters = 3;
        config.CovarianceType = CovarianceType.Full;
    })
    .ConfigurePreprocessing(config =>
    {
        config.NormalizeFeatures = true;
    })
    .BuildAsync(data);

// Get soft assignments (probabilities)
var probabilities = result.ClusterProbabilities;

Console.WriteLine("Soft Cluster Assignments:");
for (int i = 0; i < Math.Min(5, probabilities.Length); i++)
{
    Console.WriteLine($"Sample {i}: " +
        $"P(C0)={probabilities[i][0]:F3}, " +
        $"P(C1)={probabilities[i][1]:F3}, " +
        $"P(C2)={probabilities[i][2]:F3}");
}

Console.WriteLine($"\nBIC Score: {result.BicScore:F2}");
Console.WriteLine($"AIC Score: {result.AicScore:F2}");
```

## Assigning New Data Points

```csharp
using AiDotNet;

// Train clustering model
var result = await new AiModelBuilder<double, double[][], int[]>()
    .ConfigureClustering(config =>
    {
        config.Algorithm = ClusteringAlgorithm.KMeans;
        config.NumClusters = 5;
    })
    .ConfigurePreprocessing(config =>
    {
        config.NormalizeFeatures = true;
    })
    .BuildAsync(trainingData);

// Assign new customers to existing clusters
var newCustomers = new double[][]
{
    new[] { 25.0, 85.0, 90.0 },   // Young, high income, high spending
    new[] { 55.0, 45.0, 20.0 },   // Older, moderate income, low spending
};

var assignments = result.Predict(newCustomers);

for (int i = 0; i < newCustomers.Length; i++)
{
    Console.WriteLine($"New customer {i + 1}: Assigned to Cluster {assignments[i]}");
}
```

## Cluster Evaluation

```csharp
using AiDotNet;

var result = await new AiModelBuilder<double, double[][], int[]>()
    .ConfigureClustering(config =>
    {
        config.Algorithm = ClusteringAlgorithm.KMeans;
        config.NumClusters = 5;
    })
    .ConfigurePreprocessing(config =>
    {
        config.NormalizeFeatures = true;
    })
    .BuildAsync(data);

// Comprehensive evaluation metrics
Console.WriteLine("Cluster Quality Metrics:");
Console.WriteLine($"Silhouette Score: {result.SilhouetteScore:F4}");
Console.WriteLine("  (Range: -1 to 1, higher is better)");

Console.WriteLine($"\nCalinski-Harabasz Index: {result.CalinskiHarabaszIndex:F2}");
Console.WriteLine("  (Higher values indicate better defined clusters)");

Console.WriteLine($"\nDavies-Bouldin Index: {result.DaviesBouldinIndex:F4}");
Console.WriteLine("  (Lower values indicate better separation)");

Console.WriteLine($"\nInertia (WCSS): {result.Inertia:F2}");
Console.WriteLine("  (Lower values indicate tighter clusters)");
```

## Feature Importance for Clustering

```csharp
using AiDotNet;

var result = await new AiModelBuilder<double, double[][], int[]>()
    .ConfigureClustering(config =>
    {
        config.Algorithm = ClusteringAlgorithm.KMeans;
        config.NumClusters = 5;
        config.ComputeFeatureImportance = true;
    })
    .ConfigurePreprocessing(config =>
    {
        config.NormalizeFeatures = true;
    })
    .BuildAsync(data);

Console.WriteLine("Feature Importance for Clustering:");
var featureNames = new[] { "Age", "Income", "Spending Score" };
for (int i = 0; i < result.FeatureImportance.Length; i++)
{
    Console.WriteLine($"  {featureNames[i]}: {result.FeatureImportance[i]:F3}");
}
```

## Complete Customer Segmentation Pipeline

```csharp
using AiDotNet;

// Full pipeline example
var customers = LoadCustomerData();
Console.WriteLine($"Loaded {customers.Length} customers");

// Build and evaluate model
var result = await new AiModelBuilder<double, double[][], int[]>()
    .ConfigureClustering(config =>
    {
        config.Algorithm = ClusteringAlgorithm.KMeans;
        config.AutoDetectClusters = true;
        config.MinClusters = 2;
        config.MaxClusters = 10;
        config.ComputeFeatureImportance = true;
    })
    .ConfigurePreprocessing(config =>
    {
        config.NormalizeFeatures = true;
        config.HandleMissingValues = true;
    })
    .BuildAsync(customers);

// Print results
Console.WriteLine($"\n=== Segmentation Complete ===");
Console.WriteLine($"Optimal clusters: {result.NumClusters}");
Console.WriteLine($"Silhouette Score: {result.SilhouetteScore:F4}");

// Analyze each segment
Console.WriteLine("\n=== Customer Segments ===");
foreach (var cluster in result.ClusterSummaries)
{
    string segmentName = ClassifySegment(cluster.Centroid);

    Console.WriteLine($"\n{segmentName} (Cluster {cluster.Id}):");
    Console.WriteLine($"  Size: {cluster.Size} customers ({100.0 * cluster.Size / customers.Length:F1}%)");
    Console.WriteLine($"  Avg Age: {cluster.Centroid[0]:F1} years");
    Console.WriteLine($"  Avg Income: ${cluster.Centroid[1]:F1}K");
    Console.WriteLine($"  Avg Spending: {cluster.Centroid[2]:F1}");
}

// Save model for future use
result.SaveModel("customer_segmentation.aimodel");
Console.WriteLine("\nModel saved for future predictions");

string ClassifySegment(double[] centroid)
{
    var income = centroid[1];
    var spending = centroid[2];

    return (income, spending) switch
    {
        ( > 70, > 70) => "High Value",
        ( > 70, < 30) => "High Income Potential",
        ( < 30, > 70) => "Budget Enthusiasts",
        ( < 30, < 30) => "Price Sensitive",
        _ => "Average"
    };
}
```

## Best Practices

1. **Always normalize features**: Clustering algorithms are sensitive to feature scales
2. **Use multiple metrics**: No single metric tells the whole story
3. **Validate with domain knowledge**: Clusters should make business sense
4. **Try multiple algorithms**: Different algorithms may reveal different patterns
5. **Handle outliers**: Consider DBSCAN for datasets with outliers

## Summary

AiDotNet's `AiModelBuilder` provides:
- Multiple clustering algorithms (K-Means, DBSCAN, Hierarchical, GMM)
- Automatic cluster detection
- Comprehensive evaluation metrics
- Outlier detection
- Feature importance analysis
- Easy prediction for new data points

All complexity is handled internally. You focus on understanding your data.
