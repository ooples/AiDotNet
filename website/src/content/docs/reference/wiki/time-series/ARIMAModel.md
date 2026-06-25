---
title: "ARIMAModel"
description: "Implements an ARIMA (AutoRegressive Integrated Moving Average) model for time series forecasting."
section: "Reference"
---

_Time-Series Models_

Implements an ARIMA (AutoRegressive Integrated Moving Average) model for time series forecasting.

## For Beginners

ARIMA is a popular technique for analyzing and forecasting time series data (data collected over time, like stock prices, temperature readings, or monthly sales figures).

## How It Works

ARIMA models are widely used for time series forecasting. The model combines three components: - AR (AutoRegressive): Uses the dependent relationship between an observation and a number of lagged observations - I (Integrated): Uses differencing of observations to make the time series stationary - MA (Moving Average): Uses the dependency between an observation and residual errors from a moving average model 

Think of ARIMA as combining three different approaches:

AutoRegressive (AR): Looks at past values to predict future values. For example, today's temperature might be related to yesterday's temperature.Integrated (I): Transforms the data to make it easier to analyze by removing trends. For example, instead of looking at temperatures directly, we might look at how they change from day to day.Moving Average (MA): Looks at past prediction errors to improve future predictions. For example, if we consistently underestimate temperature, we can adjust for that.

The model has three key parameters (p, d, q):

p: How many past values to look at (AR component)d: How many times to difference the data (I component)q: How many past prediction errors to consider (MA component)

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
    .ConfigureModel(new ARIMAModel<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrixVector(x, new Vector<double>(series)))
    .BuildAsync();

var forecast = result.Predict(new Matrix<double>(6, 1));
Console.WriteLine($"ARIMAModel: forecast {forecast.Length} steps.");
```

