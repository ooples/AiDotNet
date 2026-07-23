# Anomaly Detection with DBSCAN

This sample demonstrates anomaly detection in transaction data using DBSCAN (Density-Based Spatial Clustering of Applications with Noise), which naturally identifies outliers as "noise" points.

## What You'll Learn

- How to use `DBSCAN<T>` for anomaly detection
- How to tune DBSCAN parameters (epsilon and MinPoints)
- How to evaluate anomaly detection with precision, recall, and F1-score
- How to analyze contamination rates and their impact on detection
- How to interpret and act on detected anomalies

## Business Context

Anomaly detection in financial transactions helps:
- **Detect fraudulent transactions** before they cause financial loss
- **Identify suspicious patterns** that may indicate account compromise
- **Reduce false positives** that frustrate legitimate customers
- **Meet regulatory requirements** for fraud monitoring

## The Dataset

This sample generates synthetic transaction data with four features:

| Feature | Description | Normal Range | Anomaly Indicators |
|---------|-------------|--------------|-------------------|
| Amount | Transaction amount in dollars | $5-$400 | >$500 unusual |
| Hour | Hour of day (0-23) | 9 AM - 10 PM | 2-5 AM suspicious |
| Day | Day of week (0-6) | Any | Combined with other factors |
| Distance | Miles from home location | 0-50 miles | >100 miles unusual |

Anomalies are generated with one or more red flags:
- **Large amounts**: Transactions $500-$10,000
- **Unusual hours**: Transactions between 2-5 AM
- **Far from home**: Transactions 200+ miles away
- **Multiple flags**: Combination of the above

## Running the Sample

```bash
cd samples/clustering/AnomalyDetection
dotnet run
```

## Expected Output

```
=== AiDotNet Anomaly Detection ===
Detecting fraudulent transactions using DBSCAN clustering

Generated 1000 transactions
True anomaly rate: 5.0%

Sample Transaction Data:
----------------------------------------------------------------
Transaction ID | Amount ($) | Hour | Day | Distance (mi) | Anomaly
----------------------------------------------------------------
   TXN00001    |     52.34  |   14 |   2 |      8.5      |
   TXN00002    |   2345.67  |    3 |   5 |    245.3      |  *
   TXN00003    |     28.91  |   11 |   1 |     12.1      |
...

=== Method 1: DBSCAN Anomaly Detection ===

Testing DBSCAN parameters...

Epsilon | MinPts | Clusters | Noise | Precision | Recall | F1-Score
--------|--------|----------|-------|-----------|--------|----------
   0.5  |     5  |      3   |    62 |    0.758  |  0.940 |   0.839
   0.7  |     5  |      2   |    48 |    0.833  |  0.800 |   0.816
   1.0  |     5  |      1   |    35 |    0.886  |  0.620 |   0.729
...

Best DBSCAN parameters: Epsilon=0.5, MinPoints=5
Best F1-Score: 0.8392

=== Contamination Rate Analysis ===

Contam. Rate | Flagged | True Pos | False Pos | Precision | Recall | F1-Score
-------------|---------|----------|-----------|-----------|--------|----------
     0.01    |     10  |       8  |        2  |    0.800  |  0.160 |   0.267
     0.02    |     20  |      18  |        2  |    0.900  |  0.360 |   0.514
     0.03    |     30  |      27  |        3  |    0.900  |  0.540 |   0.675
     0.05    |     50  |      45  |        5  |    0.900  |  0.900 |   0.900
     0.07    |     70  |      48  |       22  |    0.686  |  0.960 |   0.800
     0.10    |    100  |      50  |       50  |    0.500  |  1.000 |   0.667
     0.15    |    150  |      50  |      100  |    0.333  |  1.000 |   0.500

=== Detected Anomalies Analysis ===

Total transactions flagged as anomalies: 62
  - True Positives (correctly identified fraud): 47
  - False Positives (normal flagged as fraud): 15

Sample Detected Anomalies:
----------------------------------------------------------------
Transaction ID | Amount ($) | Hour | Day | Distance | Actual
----------------------------------------------------------------
   TXN00023    |   3456.78  |    2 |   4 |    312.5  | FRAUD
   TXN00089    |     85.23  |   14 |   1 |      5.2  | Normal
   TXN00156    |   8234.12  |   16 |   3 |     45.8  | FRAUD
...

=== Summary ===

Detection Performance:
  Precision: 0.7581 (75.8% of flagged transactions are actual fraud)
  Recall:    0.9400 (94.0% of actual frauds were detected)
  F1-Score:  0.8392 (harmonic mean of precision and recall)

Fraud Detection:
  Total actual fraudulent transactions: 50
  Frauds detected: 47
  Frauds missed: 3

=== Sample Complete ===
```

## Code Highlights

### DBSCAN Configuration

```csharp
var dbscan = new DBSCAN<double>(new DBSCANOptions<double>
{
    Epsilon = 0.5,           // Neighborhood radius
    MinPoints = 5,           // Minimum points to form dense region
    Algorithm = NeighborAlgorithm.Auto  // KDTree, BallTree, or BruteForce
});

dbscan.Train(normalizedData);
var labels = dbscan.Labels;  // -1 indicates noise (anomaly)
```

### Identifying Anomalies

```csharp
// DBSCAN labels noise points as -1
var anomalyIndices = new List<int>();
for (int i = 0; i < labels.Length; i++)
{
    if ((int)labels[i] == DBSCAN<double>.NoiseLabel)  // NoiseLabel = -1
    {
        anomalyIndices.Add(i);
    }
}
```

### Parameter Tuning

```csharp
// Grid search over epsilon and MinPoints
foreach (double eps in new[] { 0.3, 0.5, 0.7, 1.0 })
{
    foreach (int minPts in new[] { 3, 5, 10 })
    {
        var dbscan = new DBSCAN<double>(new DBSCANOptions<double>
        {
            Epsilon = eps,
            MinPoints = minPts
        });
        dbscan.Train(data);

        var (precision, recall, f1) = CalculateMetrics(trueLabels, predictedAnomalies);
        // Select parameters with best F1-score
    }
}
```

## Understanding DBSCAN for Anomaly Detection

### How DBSCAN Works

1. **Core Points**: Points with at least `MinPoints` neighbors within `Epsilon` distance
2. **Border Points**: Points within `Epsilon` of a core point but with fewer than `MinPoints` neighbors
3. **Noise Points**: Points that are neither core nor border - these are our anomalies

### Why DBSCAN for Anomaly Detection?

| Advantage | Description |
|-----------|-------------|
| **No cluster count needed** | Unlike K-Means, DBSCAN finds clusters automatically |
| **Natural outlier detection** | Noise points are inherently identified |
| **Arbitrary cluster shapes** | Can handle non-spherical normal behavior patterns |
| **Density-based** | Anomalies are rare, isolated points by definition |

### Parameter Selection

| Parameter | Too Small | Too Large |
|-----------|-----------|-----------|
| **Epsilon** | Many small clusters, more noise (high recall, low precision) | One large cluster, fewer noise points (low recall, high precision) |
| **MinPoints** | More core points, fewer anomalies | Fewer core points, more anomalies flagged |

### Rule of Thumb

- **MinPoints**: Use `2 * dimensions` as a starting point (e.g., 8 for 4 features)
- **Epsilon**: Use a k-distance graph to find the "elbow" point

## Understanding the Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Precision** | TP / (TP + FP) | Of flagged transactions, what % are actual fraud? |
| **Recall** | TP / (TP + FN) | Of actual frauds, what % did we catch? |
| **F1-Score** | 2 * P * R / (P + R) | Harmonic mean - balances precision and recall |

### Trade-offs

- **High Precision, Low Recall**: Few false alarms, but miss many frauds
- **High Recall, Low Precision**: Catch most frauds, but many false alarms
- **Balanced F1**: Good compromise for most applications

### Business Considerations

- **Banking**: May prefer high recall (catch all frauds, accept more false positives)
- **E-commerce**: May prefer balanced F1 (good detection without excessive customer friction)
- **Enterprise**: May prefer high precision (minimize manual review workload)

## Contamination Rate Analysis

The contamination rate is the expected proportion of anomalies in your data. This analysis helps you:

1. **Set decision thresholds** based on business requirements
2. **Estimate resource needs** for manual review
3. **Balance detection** against customer experience

## Alternative Approaches

### Isolation Forest

For larger datasets or when you need anomaly scores (not just binary labels), consider `TimeSeriesIsolationForest`:

```csharp
var isolationForest = new TimeSeriesIsolationForest<double>(
    new TimeSeriesIsolationForestOptions<double>
    {
        NumTrees = 100,
        ContaminationRate = 0.05,
        SampleSize = 256
    });

var scores = isolationForest.DetectAnomalies(data);
```

### HDBSCAN

For varying density clusters, use `HDBSCAN`:

```csharp
var hdbscan = new HDBSCAN<double>(new HDBSCANOptions<double>
{
    MinClusterSize = 15,
    MinSamples = 5
});
```

## Next Steps

- Try HDBSCAN for datasets with varying density patterns
- Implement real-time scoring using the trained model
- Combine with other features like merchant category, device fingerprint
- Add temporal features to detect velocity-based attacks

## Related Samples

- [CustomerSegmentation](../CustomerSegmentation/) - K-Means clustering for customer groups
- [BasicClassification](../../getting-started/BasicClassification/) - Supervised fraud detection
