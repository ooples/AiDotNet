---
title: "Time Series"
description: "Forecasting and anomaly detection."
order: 8
section: "Tutorials"
---

Learn how to forecast future values and detect anomalies in temporal data using AiDotNet.

## Overview

AiDotNet provides time series models for:
- **Forecasting**: Predict future values from historical data
- **Anomaly Detection**: Identify unusual patterns or outliers
- **Classification**: Classify temporal patterns (e.g., activity recognition)

---

## Quick Start

```csharp
using AiDotNet;
using AiDotNet.TimeSeries;

// Historical monthly sales data
var salesData = new double[]
{
    120, 135, 148, 160, 155, 170,
    180, 195, 210, 198, 220, 235,
    140, 155, 165, 178, 172, 190,
    200, 215, 230, 218, 245, 260
};

// Build ARIMA model
var result = await new AiModelBuilder<double, double[], double>()
    .ConfigureModel(new ARIMAModel<double>(p: 2, d: 1, q: 1))
    .BuildAsync(salesData);

// Forecast next 6 months
var forecast = result.Forecast(steps: 6);
foreach (var (step, value) in forecast.Select((v, i) => (i + 1, v)))
{
    Console.WriteLine($"Month +{step}: {value:F0}");
}
```

---

## Available Models

### Statistical Models

| Model | Description | Best For |
|:------|:------------|:---------|
| `ARIMAModel` | AutoRegressive Integrated Moving Average | Univariate, stationary data |
| `SARIMAModel` | Seasonal ARIMA | Data with seasonal patterns |
| `ExponentialSmoothing` | Holt-Winters | Trend + seasonality |
| `ProphetModel` | Facebook Prophet-style | Business forecasting |

### Deep Learning Models

| Model | Description | Best For |
|:------|:------------|:---------|
| `DeepARModel` | Autoregressive RNN | Probabilistic forecasting |
| `NBEATSModel` | Neural Basis Expansion | Univariate forecasting |
| `TemporalFusionTransformer` | Attention-based | Multi-horizon with covariates |
| `NHiTSModel` | Hierarchical interpolation | Long-horizon forecasting |

---

## Deep Learning Forecasting

```csharp
using AiDotNet;
using AiDotNet.TimeSeries;

// DeepAR for probabilistic forecasting
var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
    .ConfigureModel(new DeepARModel<float>(new DeepAROptions<float>()))
    .ConfigureOptimizer(new AdamOptimizer<float>(learningRate: 0.001f))
    .BuildAsync(trainSequences, trainTargets);

// Forecast next 6 steps
var forecast = result.Forecast(history, steps: 6);
```

---

## Anomaly Detection

```csharp
using AiDotNet;
using AiDotNet.TimeSeries.AnomalyDetection;

// Detect anomalies in sensor readings using Isolation Forest
var detector = new TimeSeriesIsolationForest<double>(
    new TimeSeriesIsolationForestOptions<double>
    {
        NumTrees = 100,
        ContaminationFraction = 0.05  // Expected 5% anomalies
    });

// Fit on historical data and detect anomalies
var sensorVector = Vector<double>.FromArray(sensorData);
var anomalyScores = detector.DetectAnomalies(sensorVector);
for (int i = 0; i < anomalyScores.Length; i++)
{
    if (NumOps<double>.GreaterThan(anomalyScores[i], threshold))
    {
        Console.WriteLine($"Anomaly at index {i}: score {anomalyScores[i]}");
    }
}
```

---

## Feature Engineering

### Common Time Series Features

```csharp
using AiDotNet.Preprocessing.TimeSeries;

// Lag features using LagLeadTransformer
var lagTransformer = new LagLeadTransformer<double>(
    new TimeSeriesFeatureOptions { LagSteps = new[] { 1, 7, 14, 30 } });
var lagFeatures = lagTransformer.Transform(data);

// Rolling statistics using RollingStatsTransformer
var rollingTransformer = new RollingStatsTransformer<double>(
    new TimeSeriesFeatureOptions
    {
        WindowSize = 7,
        EnabledStatistics = new[] { RollingStat.Mean, RollingStat.Std, RollingStat.Min, RollingStat.Max }
    });
var rollingFeatures = rollingTransformer.Transform(data);
```

---

## Best Practices

1. **Check stationarity**: Use ADF test before applying ARIMA
2. **Handle seasonality**: Use SARIMA or seasonal decomposition
3. **Scale your data**: Neural network models need normalized input
4. **Use walk-forward validation**: Not random train/test splits
5. **Monitor forecast horizon**: Accuracy degrades with longer horizons

---

## Next Steps

- [Regression Tutorial](/docs/tutorials/regression/) - For non-temporal prediction
- [Deployment Tutorial](/docs/tutorials/deployment/) - Serve your models in production
