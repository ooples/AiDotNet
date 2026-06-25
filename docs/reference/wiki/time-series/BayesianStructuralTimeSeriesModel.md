---
title: "BayesianStructuralTimeSeriesModel"
description: "Implements a Bayesian Structural Time Series model for flexible time series forecasting."
section: "Reference"
---

_Time-Series Models_

Implements a Bayesian Structural Time Series model for flexible time series forecasting.

## How It Works

The Bayesian Structural Time Series (BSTS) model is a powerful and flexible approach for analyzing
and forecasting time series data. It decomposes a time series into interpretable components
including level, trend, seasonality, and regression effects, using Bayesian methods to handle
uncertainty and combine information from different sources.

For Beginners:
A Bayesian Structural Time Series model is like having a flexible toolkit for forecasting time series data.
Unlike simpler models like AR or ARIMA that use fixed patterns, BSTS breaks down your data into
meaningful components:

1. Level: The current "baseline" value of your series
2. Trend: Whether your data is generally increasing or decreasing over time
3. Seasonal patterns: Regular cycles in your data (daily, weekly, yearly, etc.)
4. Effects of external factors: How other variables influence your data

The "Bayesian" part means the model handles uncertainty well. Instead of making single point predictions,
it gives you a range of possible outcomes with probabilities attached. It uses something called a
"Kalman filter" to continually update its understanding as new data arrives.

BSTS models are especially powerful because they:

- Can handle missing data
- Allow you to incorporate external information
- Provide predictions with uncertainty ranges
- Let you see which components are driving your forecast

This makes them ideal for analyzing complex time series where you want to understand
what's driving changes in addition to making predictions.

## Example

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.TimeSeries;
using AiDotNet.Tensors.LinearAlgebra;

double[] series =
{
    120, 135, 148, 160, 155, 170, 180, 195, 210, 198, 220, 235,
    140, 155, 165, 178, 172, 190, 200, 215, 230, 218, 245, 260
};
var x = new Matrix<double>(series.Length, 1);
for (int i = 0; i < series.Length; i++) x[i, 0] = i;

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new BayesianStructuralTimeSeriesModel<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrixVector(x, new Vector<double>(series)))
    .BuildAsync();

var forecast = result.Predict(new Matrix<double>(6, 1));
Console.WriteLine($"BayesianStructuralTimeSeriesModel: forecast {forecast.Length} steps.");
```

