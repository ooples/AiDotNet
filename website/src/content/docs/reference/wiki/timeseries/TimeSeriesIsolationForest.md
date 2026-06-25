---
title: "TimeSeriesIsolationForest<T>"
description: "Implements Isolation Forest for time series anomaly detection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TimeSeries.AnomalyDetection`

Implements Isolation Forest for time series anomaly detection.

## For Beginners

Imagine you're trying to describe where someone lives.
For most people, you need many questions: "Which continent? Which country? Which city?..."
But if someone lives on a tiny island, you can identify them quickly: "Do you live on that island? Yes."

Isolation Forest uses this idea: anomalies are "easy to describe" (short paths),
while normal points need more questions to distinguish them.

For time series, we add context: "Is this value unusual compared to yesterday?
Is it unusual for this time of day? Is it unusual given the recent trend?"

## How It Works

**The Time Series Anomaly Detection Challenge:**
Traditional anomaly detection treats each data point independently. For time series,
we need to consider temporal context - a value might be normal on its own but
anomalous given what came before or the time of day.

**How Time Series Isolation Forest Works:**

1. **Feature Engineering**: Transform raw time series into feature vectors including:
- Lag features (past values)
- Rolling statistics (mean, std, min, max over recent windows)
- Trend indicators (derivative, acceleration)
- Seasonal residuals (deviation from expected seasonal pattern)

2. **Isolation Forest**: For each feature vector:
- Randomly select a feature and split value
- Recursively partition until isolated
- Count path length to isolation
- Anomalies have shorter paths (easier to isolate)

3. **Anomaly Scoring**: Compute anomaly score from average path length across all trees

## Example

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.TimeSeries.AnomalyDetection;
using AiDotNet.Tensors.LinearAlgebra;

double[] series =
{
    120, 135, 148, 160, 155, 170, 180, 195, 210, 198, 220, 235,
    140, 155, 165, 178, 172, 190, 200, 215, 230, 218, 245, 260
};
var x = new Matrix<double>(series.Length, 1);
for (int i = 0; i < series.Length; i++) x[i, 0] = i;

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new TimeSeriesIsolationForest<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrixVector(x, new Vector<double>(series)))
    .BuildAsync();

var forecast = result.Predict(new Matrix<double>(6, 1));
Console.WriteLine($"TimeSeriesIsolationForest: forecast {forecast.Length} steps.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TimeSeriesIsolationForest(TimeSeriesIsolationForestOptions<>)` | Initializes a new instance of the Time Series Isolation Forest. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeC(Int32)` | Computes c(n), the average path length of unsuccessful search in BST. |
| `CreateInstance` |  |
| `DeserializeCore(BinaryReader)` |  |
| `DetectAnomalies(Vector<>)` | Detects anomalies in the time series and returns anomaly scores. |
| `GetAnomalyIndices(Vector<>)` | Gets the indices of detected anomalies. |
| `GetAnomalyLabels(Vector<>)` | Returns binary anomaly labels (true = anomaly). |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `PredictSingle(Vector<>)` |  |
| `SerializeCore(BinaryWriter)` |  |
| `TrainCore(Matrix<>,Vector<>)` | Trains the isolation forest on the time series data. |

