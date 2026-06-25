---
title: "GARCHModel"
description: "Represents a Generalized Autoregressive Conditional Heteroskedasticity (GARCH) model for time series with changing volatility."
section: "Reference"
---

_Time-Series Models_

Represents a Generalized Autoregressive Conditional Heteroskedasticity (GARCH) model for time series with changing volatility.

## For Beginners

GARCH models help predict both the value and the uncertainty of financial data. Think of it like weather forecasting: - Regular forecasting predicts the temperature (the mean value) - GARCH also predicts how much the temperature might vary (the volatility) For example, with stock prices: - Some days, prices barely change (low volatility) - Other days, prices swing wildly up and down (high volatility) - Often, volatile periods tend to cluster together (volatility clustering) GARCH helps model this behavior by: - Using one model to predict the average price (mean model) - Using another model to predict how much prices might fluctuate (volatility model) This is especially useful for financial risk management, option pricing, and trading strategies.

## How It Works

The GARCH model is specifically designed for time series data that exhibits volatility clustering, where periods of high volatility are followed by periods of high volatility, and periods of low volatility are followed by periods of low volatility. It combines a mean model (typically an ARIMA model) with a variance model that captures the conditional heteroskedasticity.

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
    .ConfigureModel(new GARCHModel<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrixVector(x, new Vector<double>(series)))
    .BuildAsync();

var forecast = result.Predict(new Matrix<double>(6, 1));
Console.WriteLine($"GARCHModel: forecast {forecast.Length} steps.");
```

