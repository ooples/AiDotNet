# Clustering Example: Customer Segmentation

This guide demonstrates how to use clustering algorithms in AiDotNet for customer segmentation.

## Overview

Customer segmentation is a classic clustering use case where we group customers based on their behavior patterns without predefined labels. This example uses K-Means and DBSCAN to segment customers.

## The Dataset

We'll work with a customer purchase dataset containing:
- Age
- Annual income
- Spending score (1-100)

```csharp
using AiDotNet;
using AiDotNet.Clustering;
using AiDotNet.Clustering.Options;

// Sample customer data: [Age, AnnualIncome($K), SpendingScore]
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
```

## Step 1: Data Preprocessing

Always scale features before clustering:

```csharp
using AiDotNet.Preprocessing;

// Scale features to zero mean and unit variance
var scaler = new StandardScaler<double>();
var scaledData = scaler.FitTransform(customers);

Console.WriteLine("Data scaled successfully");
Console.WriteLine($"Original first customer: [{string.Join(", ", customers[0])}]");
Console.WriteLine($"Scaled first customer: [{string.Join(", ", scaledData[0].Select(x => $"{x:F3}"))}]");
```

## Step 2: Finding Optimal K (Number of Clusters)

### Elbow Method

```csharp
using AiDotNet.Clustering.Evaluation;

var evaluator = new ClusteringEvaluator<double>();

// Calculate WCSS for different K values
Console.WriteLine("\nElbow Method Analysis:");
Console.WriteLine("K\tWCSS\t\tSilhouette");
Console.WriteLine("---\t----\t\t----------");

for (int k = 2; k <= 10; k++)
{
    var kmeans = new KMeans<double>(new KMeansOptions<double>
    {
        K = k,
        MaxIterations = 100,
        RandomState = 42
    });

    kmeans.Fit(scaledData);

    var wcss = evaluator.WCSS(scaledData, kmeans.Labels, kmeans.ClusterCenters);
    var silhouette = evaluator.SilhouetteScore(scaledData, kmeans.Labels);

    Console.WriteLine($"{k}\t{wcss:F2}\t\t{silhouette:F4}");
}

// Look for the "elbow" - where WCSS decrease slows down
```

### Gap Statistic

```csharp
// More rigorous method for determining optimal K
var gapResult = evaluator.GapStatistic(scaledData, kRange: Enumerable.Range(2, 9));

Console.WriteLine($"\nGap Statistic suggests K = {gapResult.OptimalK}");
Console.WriteLine($"Gap value: {gapResult.GapValues[gapResult.OptimalK - 2]:F4}");
```

## Step 3: K-Means Clustering

```csharp
// Based on elbow analysis, let's use K=5
var kmeans = new KMeans<double>(new KMeansOptions<double>
{
    K = 5,
    InitMethod = KMeansInitMethod.KMeansPlusPlus,
    MaxIterations = 300,
    Tolerance = 1e-4,
    RandomState = 42
});

// Fit the model
kmeans.Fit(scaledData);

// Get results
var labels = kmeans.Labels;
var centers = kmeans.ClusterCenters;
var iterations = kmeans.NumIterations;

Console.WriteLine($"\nK-Means converged in {iterations} iterations");
Console.WriteLine($"Final inertia (WCSS): {kmeans.Inertia:F2}");
```

## Step 4: Analyzing Clusters

```csharp
// Analyze each cluster
Console.WriteLine("\n=== Cluster Analysis ===");

for (int cluster = 0; cluster < 5; cluster++)
{
    // Get customers in this cluster
    var clusterIndices = labels
        .Select((label, idx) => new { label, idx })
        .Where(x => x.label == cluster)
        .Select(x => x.idx)
        .ToArray();

    // Calculate statistics for original (unscaled) data
    var clusterCustomers = clusterIndices.Select(i => customers[i]).ToArray();

    var avgAge = clusterCustomers.Average(c => c[0]);
    var avgIncome = clusterCustomers.Average(c => c[1]);
    var avgSpending = clusterCustomers.Average(c => c[2]);

    Console.WriteLine($"\nCluster {cluster} ({clusterIndices.Length} customers):");
    Console.WriteLine($"  Average Age: {avgAge:F1} years");
    Console.WriteLine($"  Average Income: ${avgIncome:F1}K");
    Console.WriteLine($"  Average Spending Score: {avgSpending:F1}");

    // Assign business-friendly labels
    string segmentName = (avgIncome, avgSpending) switch
    {
        ( > 70, > 70) => "High Value",
        ( > 70, < 30) => "High Income, Low Spend (Potential)",
        ( < 30, > 70) => "Budget Enthusiasts",
        ( < 30, < 30) => "Price Sensitive",
        _ => "Average"
    };

    Console.WriteLine($"  Segment: {segmentName}");
}
```

## Step 5: DBSCAN for Comparison

DBSCAN can find clusters of arbitrary shape and identify outliers:

```csharp
// Try DBSCAN
var dbscan = new DBSCAN<double>(new DBSCANOptions<double>
{
    Epsilon = 0.5,   // Neighborhood radius
    MinPoints = 5    // Minimum points to form a cluster
});

dbscan.Fit(scaledData);

var dbscanLabels = dbscan.Labels;
var numClusters = dbscanLabels.Where(l => l >= 0).Distinct().Count();
var numOutliers = dbscanLabels.Count(l => l == -1);

Console.WriteLine($"\n=== DBSCAN Results ===");
Console.WriteLine($"Number of clusters: {numClusters}");
Console.WriteLine($"Number of outliers: {numOutliers}");

// Outliers might be unusual customers worth investigating
if (numOutliers > 0)
{
    Console.WriteLine("\nOutlier customers (unusual patterns):");
    for (int i = 0; i < dbscanLabels.Length; i++)
    {
        if (dbscanLabels[i] == -1)
        {
            Console.WriteLine($"  Customer {i}: Age={customers[i][0]}, " +
                            $"Income=${customers[i][1]}K, Spending={customers[i][2]}");
        }
    }
}
```

## Step 6: Evaluating Cluster Quality

```csharp
Console.WriteLine("\n=== Cluster Quality Metrics ===");

// Silhouette Score (-1 to 1, higher is better)
var silhouette = evaluator.SilhouetteScore(scaledData, labels);
Console.WriteLine($"Silhouette Score: {silhouette:F4}");
Console.WriteLine("  (Values > 0.5 indicate good clustering)");

// Calinski-Harabasz Index (higher is better)
var ch = evaluator.CalinskiHarabaszIndex(scaledData, labels);
Console.WriteLine($"Calinski-Harabasz Index: {ch:F2}");

// Davies-Bouldin Index (lower is better)
var db = evaluator.DaviesBouldinIndex(scaledData, labels);
Console.WriteLine($"Davies-Bouldin Index: {db:F4}");
Console.WriteLine("  (Values < 1 indicate good separation)");
```

## Step 7: Predicting New Customers

```csharp
// Assign new customers to existing clusters
var newCustomers = new double[][]
{
    new[] { 25.0, 85.0, 90.0 },  // Young, high income, high spending
    new[] { 55.0, 45.0, 20.0 },  // Older, moderate income, low spending
};

Console.WriteLine("\n=== New Customer Predictions ===");

foreach (var customer in newCustomers)
{
    // Scale the new customer using the same scaler
    var scaledCustomer = scaler.Transform(new[] { customer })[0];

    // Predict cluster
    var cluster = kmeans.Predict(scaledCustomer);

    Console.WriteLine($"Customer (Age={customer[0]}, Income=${customer[1]}K, Spending={customer[2]})");
    Console.WriteLine($"  -> Assigned to Cluster {cluster}");
}
```

## Complete Example

```csharp
using AiDotNet;
using AiDotNet.Clustering;
using AiDotNet.Clustering.Options;
using AiDotNet.Clustering.Evaluation;
using AiDotNet.Preprocessing;

class CustomerSegmentation
{
    public static void Main()
    {
        // Load customer data
        var customers = LoadCustomerData();
        Console.WriteLine($"Loaded {customers.Length} customers");

        // Preprocess
        var scaler = new StandardScaler<double>();
        var scaledData = scaler.FitTransform(customers);

        // Find optimal K
        var evaluator = new ClusteringEvaluator<double>();
        int optimalK = FindOptimalK(scaledData, evaluator);
        Console.WriteLine($"Optimal K: {optimalK}");

        // Cluster
        var kmeans = new KMeans<double>(new KMeansOptions<double>
        {
            K = optimalK,
            InitMethod = KMeansInitMethod.KMeansPlusPlus,
            MaxIterations = 300,
            RandomState = 42
        });

        kmeans.Fit(scaledData);

        // Analyze
        AnalyzeClusters(customers, kmeans.Labels, optimalK);

        // Evaluate
        var silhouette = evaluator.SilhouetteScore(scaledData, kmeans.Labels);
        Console.WriteLine($"\nFinal Silhouette Score: {silhouette:F4}");

        // Export results
        ExportResults(customers, kmeans.Labels, "customer_segments.csv");
        Console.WriteLine("\nResults exported to customer_segments.csv");
    }

    static int FindOptimalK(double[][] data, ClusteringEvaluator<double> evaluator)
    {
        double maxSilhouette = -1;
        int bestK = 2;

        for (int k = 2; k <= 10; k++)
        {
            var kmeans = new KMeans<double>(new KMeansOptions<double>
            {
                K = k,
                MaxIterations = 100,
                RandomState = 42
            });

            kmeans.Fit(data);
            var silhouette = evaluator.SilhouetteScore(data, kmeans.Labels);

            if (silhouette > maxSilhouette)
            {
                maxSilhouette = silhouette;
                bestK = k;
            }
        }

        return bestK;
    }

    static void AnalyzeClusters(double[][] customers, int[] labels, int k)
    {
        Console.WriteLine("\n=== Customer Segments ===\n");

        for (int cluster = 0; cluster < k; cluster++)
        {
            var clusterCustomers = customers
                .Where((c, i) => labels[i] == cluster)
                .ToArray();

            if (clusterCustomers.Length == 0) continue;

            var avgAge = clusterCustomers.Average(c => c[0]);
            var avgIncome = clusterCustomers.Average(c => c[1]);
            var avgSpending = clusterCustomers.Average(c => c[2]);

            Console.WriteLine($"Segment {cluster + 1}: {clusterCustomers.Length} customers");
            Console.WriteLine($"  Avg Age: {avgAge:F1}");
            Console.WriteLine($"  Avg Income: ${avgIncome:F1}K");
            Console.WriteLine($"  Avg Spending: {avgSpending:F1}");
            Console.WriteLine();
        }
    }

    static void ExportResults(double[][] customers, int[] labels, string filename)
    {
        using var writer = new StreamWriter(filename);
        writer.WriteLine("Age,Income,SpendingScore,Cluster");

        for (int i = 0; i < customers.Length; i++)
        {
            writer.WriteLine($"{customers[i][0]},{customers[i][1]},{customers[i][2]},{labels[i]}");
        }
    }

    static double[][] LoadCustomerData()
    {
        // In practice, load from file or database
        return new double[][]
        {
            new[] { 19.0, 15.0, 39.0 },
            new[] { 21.0, 15.0, 81.0 },
            new[] { 20.0, 16.0, 6.0 },
            // ... more data
        };
    }
}
```

## Business Recommendations

Based on clustering results, you might:

1. **High Value Segment**: Offer loyalty programs, exclusive products
2. **Potential Segment**: Target with engagement campaigns
3. **Budget Enthusiasts**: Promote deals and discounts
4. **Price Sensitive**: Focus on value propositions

## Summary

This example demonstrated:
- Data preprocessing for clustering
- Finding optimal number of clusters
- K-Means clustering implementation
- DBSCAN for outlier detection
- Cluster evaluation metrics
- Business interpretation of results

Clustering is a powerful tool for discovering patterns in customer data without the need for labeled examples.
