# Time Series Tutorial

Learn how to forecast future values and detect anomalies in temporal data using AiDotNet.

---

## Overview

Time series analysis involves working with data points collected over time. AiDotNet provides 30+ time series models for forecasting, anomaly detection, and pattern recognition.

## Available Models

| Model | Use Case |
|:------|:---------|
| ARIMA | Classical statistical forecasting |
| SARIMA | Seasonal ARIMA |
| Prophet | Facebook's forecasting library |
| N-BEATS | Neural basis expansion |
| Temporal Fusion Transformer | State-of-the-art deep learning |
| LSTM | Long short-term memory networks |
| GRU | Gated recurrent units |
| Transformer | Attention-based forecasting |
| DeepAR | Probabilistic forecasting |

## Quick Start

```csharp
using AiDotNet;
using AiDotNet.TimeSeries;

// Sample time series data (monthly sales)
var timestamps = Enumerable.Range(0, 24).Select(i => DateTime.Now.AddMonths(-24 + i)).ToArray();
var values = new double[] { 100, 120, 115, 130, 145, 140, 155, 160, 150, 170, 180, 190,
                            195, 210, 205, 220, 235, 240, 250, 260, 255, 270, 280, 290 };

// Build and train a forecasting model
var result = await new AiModelBuilder<double, double[], double>()
    .ConfigureModel(new LSTMForecaster<double>(
        inputSize: 12,      // Use 12 months of history
        hiddenSize: 64,
        outputSize: 3       // Predict next 3 months
    ))
    .ConfigureOptimizer(new AdamOptimizer<double>())
    .BuildAsync(values);

// Forecast next 3 months
var forecast = result.Predict(values.TakeLast(12).ToArray());
Console.WriteLine($"Forecasted values: {string.Join(", ", forecast)}");
```

## Anomaly Detection

```csharp
// Detect anomalies in time series
var anomalyDetector = await new AiModelBuilder<double, double[], double>()
    .ConfigureModel(new IsolationForest<double>())
    .BuildAsync(values);

var isAnomaly = anomalyDetector.Predict(newValue);
```

## Next Steps

- [Regression Tutorial](../regression/index.md) - Predict continuous values
- [Classification Tutorial](../classification/index.md) - Predict categories
- [API Reference](../../api/index.md) - Full documentation
