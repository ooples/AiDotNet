# Customer Segmentation with K-Means Clustering

This sample demonstrates customer segmentation using K-Means clustering, a common marketing analytics use case for identifying distinct customer groups based on their characteristics.

## What You'll Learn

- How to use `KMeans<T>` for customer segmentation
- How to find the optimal number of clusters using silhouette analysis
- How to evaluate clustering quality with multiple metrics
- How to interpret cluster characteristics for business insights
- How to generate actionable marketing recommendations

## Business Context

Customer segmentation helps businesses:
- **Target marketing campaigns** to specific customer groups
- **Personalize product recommendations** based on spending patterns
- **Optimize pricing strategies** for different customer segments
- **Improve customer retention** by understanding customer needs

## The Dataset

This sample generates synthetic customer data with three features:

| Feature | Description | Range |
|---------|-------------|-------|
| Age | Customer age in years | 18-80 |
| Annual Income | Annual income in thousands of dollars | $10K-$200K |
| Spending Score | Spending propensity score (1-100) | 1-100 |

The data contains natural clusters representing different customer personas:
- **Young Trendsetters**: Young, moderate income, high spenders
- **Premium Spenders**: High income, high spending across ages
- **Established Moderates**: Middle-aged, high income, moderate spending
- **Conservative Seniors**: Older, moderate income, low spending
- **Budget Conscious**: Price-sensitive customers across all ages

## Running the Sample

```bash
cd samples/clustering/CustomerSegmentation
dotnet run
```

## Expected Output

```
=== AiDotNet Customer Segmentation ===
Customer segmentation using K-Means clustering

Generated 500 customer records
Features: Age, Annual Income ($K), Spending Score (1-100)

Sample Customer Data:
----------------------------------------------
Customer ID | Age | Income ($K) | Spending Score
----------------------------------------------
   CUST0001 |  27 |     52.3    |       78
   CUST0002 |  45 |    112.5    |       55
   CUST0003 |  62 |     58.2    |       24
   CUST0004 |  31 |    145.8    |       89
   CUST0005 |  38 |     42.1    |       18
            ...    (495 more customers)

Finding optimal number of clusters...

  K=2: Inertia=  856.32, Silhouette Score=0.3521
  K=3: Inertia=  542.18, Silhouette Score=0.4123
  K=4: Inertia=  385.64, Silhouette Score=0.4856
  K=5: Inertia=  298.42, Silhouette Score=0.5234
  K=6: Inertia=  245.87, Silhouette Score=0.4891
  K=7: Inertia=  212.35, Silhouette Score=0.4623
  K=8: Inertia=  186.42, Silhouette Score=0.4412

Optimal K based on Silhouette Score: 5

Training final K-Means model with K=5...
Converged in 12 iterations

=== Cluster Analysis ===

Cluster 0: "Young Trendsetters"
  Size: 98 customers
  Age:            Avg= 27.2, Range=[  20-  35]
  Income ($K):    Avg= 48.5, Range=[  30-  68]
  Spending Score: Avg= 78.3, Range=[  62-  98]

Cluster 1: "Premium Spenders"
  Size: 85 customers
  Age:            Avg= 32.8, Range=[  25-  42]
  Income ($K):    Avg=138.2, Range=[ 100- 178]
  Spending Score: Avg= 84.1, Range=[  70-  99]

...

=== Clustering Quality Metrics ===

Number of Clusters: 5
Total Customers: 500

Cluster Sizes:
  Cluster 0: 98 customers (19.6%)
  Cluster 1: 85 customers (17.0%)
  Cluster 2: 112 customers (22.4%)
  Cluster 3: 95 customers (19.0%)
  Cluster 4: 110 customers (22.0%)

Internal Validity Metrics:
----------------------------------------------
  Silhouette Score            :   0.5234  (Good)
  Davies-Bouldin Index        :   0.8542  (Good)
  Calinski-Harabasz Index     : 245.6821  (Good)
  Dunn Index                  :   0.3215  (Fair)
  Connectivity Index          :  32.4500  (Good)

=== Cluster Centers (Original Scale) ===

Cluster |   Age   | Income ($K) | Spending Score
--------|---------|-------------|---------------
   0    |   27.2  |      48.5   |      78.3
   1    |   32.8  |     138.2   |      84.1
   2    |   52.4  |     105.6   |      52.3
   3    |   63.5  |      58.4   |      25.8
   4    |   45.2  |      38.6   |      15.2

=== Marketing Insights ===

Cluster 0 - Young Trendsetters:
  -> Leverage social media and influencer marketing
  -> Trendy products and limited editions appeal to this group
  -> Mobile-first engagement strategy recommended

Cluster 1 - Premium Spenders:
  -> Target with premium product launches and exclusive offers
  -> High-value customers - prioritize retention programs
  -> Offer VIP membership and early access privileges

...

=== Sample Complete ===
```

## Code Highlights

### K-Means Configuration

```csharp
var kmeans = new KMeans<double>(new KMeansOptions<double>
{
    NumClusters = 5,              // Number of segments to find
    MaxIterations = 300,          // Maximum iterations
    Tolerance = 1e-4,             // Convergence threshold
    RandomState = 42,             // For reproducibility
    NumInitializations = 10,      // Run 10 times, keep best
    InitMethod = KMeansInitMethod.KMeansPlusPlus  // Smart initialization
});

kmeans.Train(normalizedFeatures);
var labels = kmeans.Labels;
```

### Finding Optimal K

```csharp
for (int k = 2; k <= 8; k++)
{
    var kmeans = new KMeans<double>(new KMeansOptions<double> { NumClusters = k });
    kmeans.Train(data);

    var silhouetteCalculator = new SilhouetteScore<double>();
    double silhouette = silhouetteCalculator.Compute(data, kmeans.Labels!);

    // Higher silhouette score = better clustering
}
```

### Comprehensive Evaluation

```csharp
var evaluator = new ClusteringEvaluator<double>();
var result = evaluator.EvaluateAll(data, labels);

Console.WriteLine($"Silhouette Score: {result.InternalMetrics["Silhouette Score"]}");
Console.WriteLine($"Davies-Bouldin Index: {result.InternalMetrics["Davies-Bouldin Index"]}");
```

## Understanding the Metrics

| Metric | Range | Interpretation |
|--------|-------|----------------|
| **Silhouette Score** | -1 to 1 | Higher is better. >0.5 is good, >0.7 is excellent |
| **Davies-Bouldin Index** | 0 to inf | Lower is better. <1.0 indicates good separation |
| **Calinski-Harabasz Index** | 0 to inf | Higher is better. Measures between/within cluster variance |
| **Dunn Index** | 0 to inf | Higher is better. Ratio of min inter-cluster to max intra-cluster distance |
| **Connectivity Index** | 0 to inf | Lower is better. Measures how well connected cluster members are |

## Key Concepts

### Why Normalize Features?

Without normalization, features with larger scales (like income in thousands) would dominate the distance calculations over features with smaller scales (like spending score 1-100). StandardScaler ensures all features contribute equally.

### K-Means++ Initialization

K-Means++ spreads initial cluster centers apart, which:
- Reduces the chance of poor convergence
- Typically requires fewer iterations
- Produces more consistent results

### Silhouette Analysis

The silhouette score measures:
- **a(i)**: Average distance to other points in the same cluster
- **b(i)**: Average distance to points in the nearest other cluster
- **s(i) = (b(i) - a(i)) / max(a(i), b(i))**

Points with high silhouette scores are well-matched to their cluster.

## Next Steps

- Try different distance metrics (Manhattan, Cosine) for different use cases
- Experiment with other clustering algorithms like DBSCAN or GMM
- Apply to real customer data from your CRM or analytics platform
- Combine with other features like purchase history or browsing behavior

## Related Samples

- [AnomalyDetection](../AnomalyDetection/) - Detect outliers in transaction data
- [BasicClassification](../../getting-started/BasicClassification/) - Supervised learning comparison
