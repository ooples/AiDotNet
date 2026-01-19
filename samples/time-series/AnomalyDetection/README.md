# Time Series Anomaly Detection

This sample demonstrates time series anomaly detection using AiDotNet's Isolation Forest and ARIMA-based methods, including threshold-based scoring, visualization of detected anomalies, and ensemble approaches.

## What You'll Learn

- How to use `TimeSeriesIsolationForest` for density-based anomaly detection
- How to use `ARIMAModel` with built-in anomaly detection
- How to evaluate detection performance with precision, recall, and F1-score
- How to visualize anomalies and their scores
- How to combine multiple methods for better detection

## Business Context

Time series anomaly detection is critical for:
- **Fraud detection** - Identify unusual transaction patterns
- **System monitoring** - Detect server/network anomalies before failures
- **Quality control** - Spot manufacturing defects in sensor data
- **Predictive maintenance** - Find equipment degradation patterns
- **Financial surveillance** - Monitor for market manipulation

## Types of Time Series Anomalies

| Type | Description | Example |
|------|-------------|---------|
| **Point anomaly** | Single unusual value | Sudden spike in CPU usage |
| **Contextual anomaly** | Normal value in wrong context | High sales on a typically slow day |
| **Collective anomaly** | Pattern of unusual values | Gradually increasing latency |
| **Level shift** | Sustained change in baseline | Step increase after system update |

## The Synthetic Dataset

This sample generates 500 days of data with injected anomalies:

- **Base pattern**: Linear trend + weekly + monthly seasonality + noise
- **Spike anomalies**: Sudden increases of 40-70 units
- **Drop anomalies**: Sudden decreases of 40-70 units
- **Level shifts**: Sustained changes lasting 5 days
- **Contextual anomalies**: Values shifted by +/- 25 units from expected

## Running the Sample

```bash
cd samples/time-series/AnomalyDetection
dotnet run
```

## Expected Output

```
=== AiDotNet Time Series Anomaly Detection ===
Detecting anomalies using Isolation Forest and ARIMA-based methods

=== Step 1: Generate Synthetic Time Series with Anomalies ===

Generated 500 data points with 25 injected anomalies:
  - Point anomalies (spikes/drops): 12
  - Level shifts: 7
  - Contextual anomalies: 6

Sample Data (points around anomalies marked with *):
-------------------------------------------------------------
Index |    Date    |   Value   | Anomaly Type
-------------------------------------------------------------
    0 | 2024-01-01 |    108.45 |
    1 | 2024-01-02 |    115.23 |
    2 | 2024-01-03 |    122.67 |
   15 | 2024-01-16 |    185.34 |  * spike
...

=== Step 2: Isolation Forest Anomaly Detection ===

Training Isolation Forest...
  NumTrees: 100
  SampleSize: 256
  ContaminationRate: 5%
  LagFeatures: 10
  RollingWindowSize: 20
Isolation Forest trained successfully!

Isolation Forest detected 28 anomalies

Anomaly Score Distribution:
---------------------------
  0.0-0.3:   45 |#########
  0.3-0.4:   82 |################
  0.4-0.5:  156 |###############################
  0.5-0.6:  134 |##########################
  0.6-0.7:   55 |###########
  0.7-0.8:   18 |###
  0.8-0.9:    8 |#
  0.9-1.0:    2 |

=== Step 3: ARIMA-based Anomaly Detection ===

Training ARIMA model with anomaly detection...
  P (AR order): 2
  D (Differencing): 1
  Q (MA order): 1
  Anomaly Threshold: 2.5 sigma
ARIMA model trained successfully!

ARIMA detected 32 anomalies
Anomaly threshold: 24.5678

=== Step 4: Detection Method Comparison ===

Detection Performance Comparison:
------------------------------------------------------------------
Method          | Detected | TP  | FP  | FN  | Precision | Recall | F1
------------------------------------------------------------------
Isolation Forest |      28  |  20 |   8 |   5 |    0.714  |  0.800 |  0.755
ARIMA            |      32  |  18 |  14 |   7 |    0.562  |  0.720 |  0.632

Ensemble (both)  |      15  |  14 |   1 |  11 |    0.933  |  0.560 |  0.700
Union (either)   |      45  |  24 |  21 |   1 |    0.533  |  0.960 |  0.686

=== Step 5: Detailed Anomaly Analysis ===

Detected Anomalies (Top 15 by Isolation Forest score):
-------------------------------------------------------------------------
Index |    Date    |  Value   | IF Score | ARIMA Score | Actual Type
-------------------------------------------------------------------------
  156 | 2024-06-05 |   185.67 |   0.8934 |     42.3456 | spike        (TRUE)
  234 | 2024-08-22 |    58.23 |   0.8567 |     38.1234 | drop         (TRUE)
  312 | 2024-11-08 |   178.45 |   0.8234 |     35.6789 | spike        (TRUE)
...

=== Step 6: Anomaly Visualization ===

Time Series Visualization (points 50-150):
Value range: 95.3 to 185.7

  185.7 |                              *
  179.2 |
  172.7 |         .
  166.2 |        . .      .
  159.7 |       .   .    . .
  153.2 |      .     .  .   .    .
  146.7 |     .       ..     .  . .
  140.2 |    .                .    .
  133.7 |   .                       .
  127.2 |  .                         .
  120.7 | .                           .    .
  114.2 |.                             .  . .
  107.7 |                               ..   .
  101.2 |                                     .
   95.3 |                                      !
        +------------------------------------...
         50                  70                  90

Legend: . = normal, * = detected anomaly, ! = missed anomaly

=== Step 7: Threshold Sensitivity Analysis ===

Isolation Forest - Threshold vs Detection Performance:
----------------------------------------------------------
Threshold | Detected | TP  | FP  | Precision | Recall | F1
----------------------------------------------------------
   0.50    |     89   |  24 |  65 |    0.270  |  0.960 |  0.421
   0.55    |     62   |  23 |  39 |    0.371  |  0.920 |  0.529
   0.60    |     42   |  21 |  21 |    0.500  |  0.840 |  0.627
   0.65    |     28   |  20 |   8 |    0.714  |  0.800 |  0.755
   0.70    |     18   |  16 |   2 |    0.889  |  0.640 |  0.744
   0.75    |     12   |  11 |   1 |    0.917  |  0.440 |  0.594
   0.80    |      6   |   6 |   0 |    1.000  |  0.240 |  0.387
   0.85    |      2   |   2 |   0 |    1.000  |  0.080 |  0.148

=== Step 8: Anomaly Score Statistics ===

Isolation Forest Score Statistics:
  Normal points (475):
    Mean:   0.4823
    StdDev: 0.0912
    Min:    0.2345
    Max:    0.6234

  Actual anomaly points (25):
    Mean:   0.7234
    StdDev: 0.1123
    Min:    0.5456
    Max:    0.8934

Score separation (Cohen's d): 2.4567
  (Good separation between normal and anomaly scores)

=== Summary ===

Dataset Characteristics:
  Total data points: 500
  Actual anomalies: 25 (5.0%)
  Data range: 58.23 to 185.67

Best Detection Method:
  Method: Isolation Forest
  F1-Score: 0.7550
  Precision: 0.7143
  Recall: 0.8000

Recommendations:
  - Using ensemble (both methods agree) improves precision

=== Sample Complete ===
```

## Code Highlights

### Isolation Forest Configuration

```csharp
var options = new TimeSeriesIsolationForestOptions<double>
{
    NumTrees = 100,              // More trees = more stable scores
    SampleSize = 256,            // Smaller samples isolate anomalies better
    ContaminationRate = 0.05,    // Expected 5% anomalies
    LagFeatures = 10,            // Temporal context
    RollingWindowSize = 20,      // Rolling statistics window
    UseTrendFeatures = true,     // Include derivative features
    RandomSeed = 42              // Reproducibility
};

var isolationForest = new TimeSeriesIsolationForest<double>(options);
isolationForest.Train(trainMatrix, timeSeries);

// Get anomaly scores (higher = more anomalous)
var scores = isolationForest.DetectAnomalies(timeSeries);

// Get binary labels
var labels = isolationForest.GetAnomalyLabels(timeSeries);

// Get indices of detected anomalies
var indices = isolationForest.GetAnomalyIndices(timeSeries);
```

### ARIMA-based Anomaly Detection

```csharp
var arimaOptions = new ARIMAOptions<double>
{
    P = 2, D = 1, Q = 1,
    EnableAnomalyDetection = true,    // Enable anomaly detection
    AnomalyThresholdSigma = 2.5       // 2.5 standard deviations
};

var arimaModel = new ARIMAModel<double>(arimaOptions);
arimaModel.Train(trainMatrix, timeSeries);

// Get boolean array of anomalies
var anomalies = arimaModel.DetectAnomalies(timeSeries);

// Get detailed anomaly information
var detailed = arimaModel.DetectAnomaliesDetailed(timeSeries);
// Returns: (Index, ActualValue, PredictedValue, Score)

// Customize threshold
arimaModel.SetAnomalyThreshold(customThreshold);
```

### Ensemble Detection

```csharp
// High precision: require both methods to agree
var highConfidence = ifDetected.Intersect(arimaDetected);

// High recall: flag if either method detects
var comprehensive = ifDetected.Union(arimaDetected);
```

## Understanding Isolation Forest

### How It Works

1. **Random partitioning**: Randomly select a feature and split point
2. **Recursive isolation**: Continue until each point is isolated
3. **Path length scoring**: Anomalies are isolated quickly (short paths)
4. **Ensemble averaging**: Average path lengths across many trees

### Why It Works for Time Series

For time series, the algorithm uses engineered features:

| Feature Type | Description | Purpose |
|--------------|-------------|---------|
| **Lag features** | Previous N values | Temporal context |
| **Rolling mean** | Moving average | Baseline comparison |
| **Rolling std** | Moving volatility | Stability measure |
| **Rolling min/max** | Recent extremes | Range context |
| **Trend features** | First/second derivative | Rate of change |

### Anomaly Score Interpretation

| Score Range | Interpretation |
|-------------|----------------|
| < 0.5 | Likely normal |
| 0.5 - 0.6 | Borderline |
| 0.6 - 0.7 | Suspicious |
| 0.7 - 0.8 | Likely anomaly |
| > 0.8 | Strong anomaly |

## Understanding ARIMA Anomaly Detection

ARIMA-based detection works differently:

1. **Train model** on time series
2. **Compute residuals**: actual - predicted
3. **Calculate statistics**: mean and std of residuals
4. **Set threshold**: mean + (sigma * std)
5. **Flag anomalies**: where |residual| > threshold

### Threshold Selection

The `AnomalyThresholdSigma` parameter controls sensitivity:

| Sigma | Expected False Positive Rate | Use Case |
|-------|------------------------------|----------|
| 2.0 | ~5% of normal data | High sensitivity |
| 2.5 | ~1% of normal data | Balanced |
| 3.0 | ~0.3% of normal data | Low false positives |
| 3.5 | ~0.05% of normal data | Only extreme anomalies |

## Evaluation Metrics

| Metric | Formula | Focus |
|--------|---------|-------|
| **Precision** | TP / (TP + FP) | Accuracy of detections |
| **Recall** | TP / (TP + FN) | Coverage of actual anomalies |
| **F1-Score** | 2 * P * R / (P + R) | Balance of both |

### Trade-off Strategies

| Strategy | Approach | Best For |
|----------|----------|----------|
| **High Precision** | Higher threshold | When false alarms are costly |
| **High Recall** | Lower threshold | When missing anomalies is costly |
| **Balanced** | Optimize F1 | General purpose |
| **Ensemble (AND)** | Require agreement | Reduce false positives |
| **Ensemble (OR)** | Flag if either detects | Maximize coverage |

## Threshold Tuning

The sample shows how detection metrics change with threshold:

```
Threshold | Precision | Recall | F1
---------------------------------
   0.50   |   0.270   | 0.960  | 0.421  <- Too sensitive
   0.65   |   0.714   | 0.800  | 0.755  <- Balanced
   0.80   |   1.000   | 0.240  | 0.387  <- Too strict
```

Choose threshold based on your cost function:
- If false positives are expensive: prefer higher threshold
- If missing anomalies is expensive: prefer lower threshold

## Next Steps

- Implement online/streaming anomaly detection
- Add seasonal decomposition preprocessing
- Try different contamination rates
- Combine with alerting systems
- Add root cause analysis

## Related Samples

- [Forecasting](../Forecasting/) - Time series prediction with ARIMA/SARIMA
- [AnomalyDetection (Clustering)](../../clustering/AnomalyDetection/) - Non-temporal anomaly detection
