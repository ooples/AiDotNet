---
title: "TimeSeriesRegression"
description: "Represents a time series regression model that incorporates temporal dependencies, trends, and seasonality."
section: "Reference"
---

_Regression Models_

Represents a time series regression model that incorporates temporal dependencies, trends, and seasonality.

## For Beginners

This class helps predict future values based on patterns in time-based data. Think of it like weather forecasting: - It looks at past weather patterns to predict future weather - It can recognize long-term trends (like gradual warming) - It can detect seasonal patterns (like winter being colder than summer) - It accounts for how recent weather affects tomorrow's weather This is useful for any data that changes over time, such as stock prices, website traffic, energy consumption, or sales figures.

## How It Works

The TimeSeriesRegression class extends basic regression by accounting for the temporal structure of the data. It can model autoregressive components (past values affecting future values), trend components (long-term directional movement), and seasonal components (recurring patterns at fixed intervals).

## Example

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;

double[][] features =
{
    new[] { 1.0, 2.0 }, new[] { 2.0, 3.0 }, new[] { 3.0, 4.0 },
    new[] { 4.0, 5.0 }, new[] { 5.0, 6.0 }, new[] { 6.0, 7.0 }
};
double[] targets = { 3.0, 5.0, 7.0, 9.0, 11.0, 13.0 };

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new TimeSeriesRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained TimeSeriesRegression.");
```

