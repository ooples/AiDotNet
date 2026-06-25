---
title: "MAModel"
description: "Implements a Moving Average (MA) model for time series forecasting."
section: "Reference"
---

_Time-Series Models_

Implements a Moving Average (MA) model for time series forecasting.

## For Beginners

A Moving Average (MA) model predicts future values based on past prediction errors. Think of it like this: If you've been consistently underestimating or overestimating values in the past, the MA model learns from these mistakes and adjusts future predictions. For example, if a weather forecast has been consistently underestimating temperatures by 2 degrees for several days, an MA model would learn this pattern and adjust its future predictions upward. The key parameter of an MA model is 'q', which determines how many past prediction errors to consider. For instance, with q=3, the model looks at errors from the last three periods when making a new prediction.

## How It Works

MA models predict future values based on past prediction errors (residuals). The model is defined as: Yt = μ + et + ?1et-1 + ?2et-2 + ... + ?qet-q where Yt is the value at time t, μ is the mean, et is the error term at time t, and ?i are the MA coefficients.

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
    .ConfigureModel(new MAModel<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrixVector(x, new Vector<double>(series)))
    .BuildAsync();

var forecast = result.Predict(new Matrix<double>(6, 1));
Console.WriteLine($"MAModel: forecast {forecast.Length} steps.");
```

