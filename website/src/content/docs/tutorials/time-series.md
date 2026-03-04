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

### Neural Network Models

| Model | Description | Best For |
|:------|:------------|:---------|
| `LSTMForecaster` | Long Short-Term Memory | Complex temporal dependencies |
| `GRUForecaster` | Gated Recurrent Unit | Faster alternative to LSTM |
| `TransformerForecaster` | Temporal Transformer | Long-range dependencies |
| `TCNForecaster` | Temporal Convolutional Network | Parallel processing |

---

## Neural Network Forecasting

```csharp
var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
    .ConfigureModel(new LSTMForecaster<float>(
        inputSize: 1,
        hiddenSize: 64,
        numLayers: 2,
        lookback: 24))
    .ConfigureOptimizer(new AdamOptimizer<float>(learningRate: 0.001f))
    .BuildAsync(trainSequences, trainTargets);
```

---

## Anomaly Detection

```csharp
using AiDotNet.TimeSeries;

// Detect anomalies in sensor readings
var detector = new TimeSeriesAnomalyDetector<double>(
    method: AnomalyMethod.IsolationForest,
    contamination: 0.05);  // Expected 5% anomalies

var anomalies = detector.Detect(sensorData);
foreach (var anomaly in anomalies)
{
    Console.WriteLine($"Anomaly at index {anomaly.Index}: {anomaly.Value} (score: {anomaly.Score:F3})");
}
```

---

## Feature Engineering

### Common Time Series Features

```csharp
// Lag features
var lagged = TimeSeries.CreateLagFeatures(data, lags: new[] { 1, 7, 14, 30 });

// Rolling statistics
var rolling = TimeSeries.CreateRollingFeatures(data, window: 7,
    stats: new[] { RollingStat.Mean, RollingStat.Std, RollingStat.Min, RollingStat.Max });

// Calendar features
var calendar = TimeSeries.CreateCalendarFeatures(dates,
    features: new[] { CalendarFeature.DayOfWeek, CalendarFeature.Month, CalendarFeature.IsHoliday });
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
