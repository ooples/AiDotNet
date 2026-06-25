---
title: "VARMAModel"
description: "Implements a Vector Autoregressive Moving Average (VARMA) model for multivariate time series forecasting."
section: "Reference"
---

_Time-Series Models_

Implements a Vector Autoregressive Moving Average (VARMA) model for multivariate time series forecasting.

## For Beginners

A VARMA model helps you forecast multiple related time series at once, accounting for: - How each variable depends on its own past values - How each variable depends on other variables' past values - How past prediction errors affect current values For example, if you're analyzing economic data, a VARMA model could simultaneously forecast: - GDP growth - Unemployment rate - Inflation rate While accounting for how these variables affect each other and incorporating information from past prediction errors to improve accuracy. Think of it as a sophisticated forecasting system that recognizes both the interconnections between different variables and learns from its own mistakes.

## How It Works

The VARMA model extends the Vector Autoregressive (VAR) model by incorporating Moving Average (MA) terms, allowing it to capture more complex dynamics in multivariate time series data. It models the relationships between multiple time series variables and their past values, as well as past error terms.

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
    .ConfigureModel(new VARMAModel<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrixVector(x, new Vector<double>(series)))
    .BuildAsync();

var forecast = result.Predict(new Matrix<double>(6, 1));
Console.WriteLine($"VARMAModel: forecast {forecast.Length} steps.");
```

