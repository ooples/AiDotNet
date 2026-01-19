# Time Series Samples

This directory contains examples of time series models in AiDotNet.

## Available Samples

| Sample | Description |
|--------|-------------|
| [Forecasting](./Forecasting/) | Multi-step ahead forecasting |
| [AnomalyDetection](./AnomalyDetection/) | Detect anomalies in time series |

## Quick Start

```csharp
using AiDotNet;
using AiDotNet.TimeSeries;

var forecaster = new TemporalFusionTransformer<float>(
    inputSize: 1,
    hiddenSize: 64,
    numLayers: 2,
    horizon: 24);

var historicalData = LoadTimeSeriesData();
forecaster.Train(historicalData);

var forecast = forecaster.Predict(steps: 24);
```

## Time Series Models (30+)

### Statistical
- ARIMA / SARIMA
- Exponential Smoothing
- Prophet

### Deep Learning
- LSTM
- GRU
- Temporal Convolutional Networks
- Transformer
- N-BEATS
- Temporal Fusion Transformer

### Anomaly Detection
- Isolation Forest
- Autoencoder
- LSTM-Autoencoder

## Learn More

- [Time Series Tutorial](/docs/tutorials/time-series/)
- [API Reference](/api/AiDotNet.TimeSeries/)
