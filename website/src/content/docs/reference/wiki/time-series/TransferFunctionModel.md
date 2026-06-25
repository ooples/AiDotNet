---
title: "TransferFunctionModel"
description: "Implements a Transfer Function Model for time series analysis, which combines ARIMA modeling with external input variables to capture dynamic relationships between multiple time series."
section: "Reference"
---

_Time-Series Models_

Implements a Transfer Function Model for time series analysis, which combines ARIMA modeling with
external input variables to capture dynamic relationships between multiple time series.

## For Beginners

A Transfer Function Model helps you understand how one time series affects another over time.

For example, you might want to know:

- How advertising spending affects sales (with delays of days or weeks)
- How temperature changes affect energy consumption
- How interest rate changes impact housing prices

This model captures both:

- The internal patterns of your target variable (like sales following their own seasonal patterns)
- The external influence of input variables (like how advertising boosts sales)

It's particularly useful when you know there are external factors influencing your target variable
and you want to quantify their effects, including any time delays in those effects.

## How It Works

The Transfer Function Model extends traditional ARIMA models by incorporating the effects of
external input variables. It models the relationship between an output time series and one or more
input time series, accounting for both immediate and lagged effects.

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
    .ConfigureModel(new TransferFunctionModel<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrixVector(x, new Vector<double>(series)))
    .BuildAsync();

var forecast = result.Predict(new Matrix<double>(6, 1));
Console.WriteLine($"TransferFunctionModel: forecast {forecast.Length} steps.");
```

