---
title: "SARIMAModel"
description: "Implements a Seasonal Autoregressive Integrated Moving Average (SARIMA) model for time series forecasting."
section: "Reference"
---

_Time-Series Models_

Implements a Seasonal Autoregressive Integrated Moving Average (SARIMA) model for time series forecasting.

## For Beginners

SARIMA is used to predict future values in a time series (data collected over time) that has seasonal patterns.
Think of it like predicting ice cream sales throughout the year - there's a general trend (maybe increasing over years)
and a seasonal pattern (higher in summer, lower in winter). SARIMA can capture both these patterns.

The model has several parameters:

- p: How many previous values influence the current value
- d: How many times we need to subtract consecutive values to make the data stable
- q: How many previous prediction errors influence the current prediction
- P, D, Q: The same as above, but for seasonal patterns
- m: The length of the seasonal cycle (e.g., 12 for monthly data with yearly patterns)

## How It Works

The SARIMA model extends the ARIMA model by incorporating seasonal components, making it suitable for 
data with seasonal patterns. It combines autoregressive (AR), integrated (I), and moving average (MA) 
components for both seasonal and non-seasonal parts of the time series.

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
    .ConfigureModel(new SARIMAModel<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrixVector(x, new Vector<double>(series)))
    .BuildAsync();

var forecast = result.Predict(new Matrix<double>(6, 1));
Console.WriteLine($"SARIMAModel: forecast {forecast.Length} steps.");
```

