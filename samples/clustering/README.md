# Clustering Samples

This directory contains examples of clustering algorithms in AiDotNet.

## Available Samples

| Sample | Description |
|--------|-------------|
| [CustomerSegmentation](./CustomerSegmentation/) | Segment customers for marketing |
| [AnomalyDetection](./AnomalyDetection/) | Detect outliers using clustering |

## Quick Start

```csharp
using AiDotNet;
using AiDotNet.Clustering;

var features = new double[][] { /* data points */ };

var result = await new PredictionModelBuilder<double, double[], int>()
    .ConfigureModel(new KMeansClustering<double>(k: 5))
    .ConfigurePreprocessing()
    .BuildAsync(features);

var clusterLabels = result.Model.Predict(features);
```

## Available Clustering Algorithms (20+)

### Partition-Based
- K-Means
- K-Means++
- Mini-Batch K-Means
- K-Medoids

### Density-Based
- DBSCAN
- HDBSCAN
- OPTICS
- Mean Shift

### Hierarchical
- Agglomerative Clustering
- Divisive Clustering
- BIRCH

### Model-Based
- Gaussian Mixture Models
- Bayesian Gaussian Mixture

### Spectral
- Spectral Clustering
- Normalized Cuts

## Choosing an Algorithm

| Scenario | Recommended Algorithm |
|----------|----------------------|
| Known number of clusters | K-Means |
| Unknown clusters | DBSCAN, HDBSCAN |
| Hierarchical structure | Agglomerative |
| Varying densities | HDBSCAN |
| Large datasets | Mini-Batch K-Means |

## Learn More

- [Clustering Tutorial](/docs/tutorials/clustering/)
- [API Reference](/api/AiDotNet.Clustering/)
