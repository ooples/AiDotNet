---
title: "ARIMAXModel"
description: "Implements an ARIMAX (AutoRegressive Integrated Moving Average with eXogenous variables) model for time series forecasting."
section: "Reference"
---

_Time-Series Models_

Implements an ARIMAX (AutoRegressive Integrated Moving Average with eXogenous variables) model for time series forecasting.

## How It Works

ARIMAX extends the ARIMA model by including external (exogenous) variables that might influence the time series. The model combines: - AR (AutoRegressive): Uses the dependent relationship between an observation and lagged observations - I (Integrated): Uses differencing to make the time series stationary - MA (Moving Average): Uses the dependency between an observation and residual errors - X (eXogenous): Incorporates external variables that may influence the time series 

For Beginners: ARIMAX is an advanced technique for forecasting time series data (data collected over time like daily temperatures, stock prices, or monthly sales) that takes into account both the history of the series itself AND external factors that might influence it. Think of it like this: - Basic forecasting might just look at past sales to predict future sales - ARIMAX also considers things like holidays, promotions, or economic indicators that might affect sales The model has four components: 1. AutoRegressive (AR): Uses past values of the series itself (like yesterday's temperature to predict today's) 2. Integrated (I): Transforms the data by looking at differences between values to remove trends 3. Moving Average (MA): Looks at past prediction errors to improve future predictions 4. eXogenous (X): Includes external factors that might affect the series (like whether it's a holiday) The "X" is what makes ARIMAX different from ARIMA - it can include information from outside the time series itself.

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
    .ConfigureModel(new ARIMAXModel<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrixVector(x, new Vector<double>(series)))
    .BuildAsync();

var forecast = result.Predict(new Matrix<double>(6, 1));
Console.WriteLine($"ARIMAXModel: forecast {forecast.Length} steps.");
```

