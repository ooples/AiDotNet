---
title: "VectorAutoRegressionModel"
description: "Implements a Vector Autoregression (VAR) model for multivariate time series forecasting."
section: "Reference"
---

_Time-Series Models_

Implements a Vector Autoregression (VAR) model for multivariate time series forecasting.

## For Beginners

A VAR model helps you forecast multiple related time series at once, accounting for how they
influence each other.

For example, if you're analyzing economic data, a VAR model could simultaneously forecast:

- GDP growth
- Unemployment rate
- Inflation rate

While accounting for relationships like:

- How unemployment affects future GDP
- How GDP affects future inflation
- How each variable's past values affect its own future

Think of it as a system that recognizes the interconnected nature of multiple time series
and uses these connections to make better forecasts for all variables simultaneously.

## How It Works

The Vector Autoregression (VAR) model is a multivariate extension of the univariate autoregressive model.
It captures linear dependencies among multiple time series variables, where each variable is modeled as
a function of past values of itself and past values of other variables in the system.

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
    .ConfigureModel(new VectorAutoRegressionModel<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrixVector(x, new Vector<double>(series)))
    .BuildAsync();

var forecast = result.Predict(new Matrix<double>(6, 1));
Console.WriteLine($"VectorAutoRegressionModel: forecast {forecast.Length} steps.");
```

