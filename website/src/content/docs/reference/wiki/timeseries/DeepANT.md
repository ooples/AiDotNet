---
title: "DeepANT<T>"
description: "Implements DeepANT (Deep Learning for Anomaly Detection in Time Series)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TimeSeries.AnomalyDetection`

Implements DeepANT (Deep Learning for Anomaly Detection in Time Series).

## For Beginners

DeepANT learns what "normal" looks like in your time series,
then flags anything unusual as an anomaly. It works by:

1. Learning to predict the next value based on past values
2. Comparing actual values to predictions
3. Marking large prediction errors as anomalies

Think of it like a system that learns your daily routine - if you suddenly do something
very different, it notices and flags it as unusual.

## How It Works

DeepANT is a deep learning-based approach for unsupervised anomaly detection in time series.
It uses a convolutional neural network to learn normal patterns and identifies anomalies
as data points that deviate significantly from the learned patterns.

Key features:

- Time series prediction using CNN
- Anomaly detection based on prediction error
- Unsupervised learning (no labeled anomalies needed)
- Effective for both point anomalies and contextual anomalies

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
    .ConfigureModel(new DeepANT<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrixVector(x, new Vector<double>(series)))
    .BuildAsync();

var forecast = result.Predict(new Matrix<double>(6, 1));
Console.WriteLine($"DeepANT: forecast {forecast.Length} steps.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DeepANT(DeepANTOptions<>)` | Initializes a new instance of the DeepANT class. |
| `DeepANT(DeepANTOptions<>,Boolean)` | Private constructor for proper options instance management. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGradients(,Int32)` | Applies accumulated gradients with SGD. |
| `ComputeAndAccumulateGradients(Tensor<>,)` | Computes analytical gradients for the FC layer and accumulates them. |
| `ComputeAnomalyScores(Vector<>)` | Computes anomaly scores for a time series. |
| `ComputeAnomalyThreshold(List<>)` | Computes anomaly threshold based on training prediction errors. |
| `DetectAnomalies(Vector<>)` | Detects anomalies in a time series. |
| `ForwardWithCache(Vector<>)` | Forward pass that returns both prediction and cached features for backpropagation. |
| `GetOptions` |  |

