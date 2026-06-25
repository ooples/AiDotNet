---
title: "STLDecomposition"
description: "Implements Seasonal-Trend decomposition using LOESS (STL) for time series analysis."
section: "Reference"
---

_Time-Series Models_

Implements Seasonal-Trend decomposition using LOESS (STL) for time series analysis.

## For Beginners

STL decomposition is like breaking down a song into its basic elements - the melody (trend),
the repeating chorus (seasonal pattern), and the unique variations (residuals).

For example, if you analyze monthly sales data:

- The trend component shows the long-term increase or decrease in sales
- The seasonal component shows regular patterns that repeat (like higher sales during holidays)
- The residual component shows what's left after removing trend and seasonality (like unexpected events)

This decomposition helps you understand what's driving your time series and can improve forecasting.
The model offers different algorithms (standard, robust, and fast) to handle various types of data.

## How It Works

STL decomposition breaks down a time series into three components: trend, seasonal, and residual.
It uses locally weighted regression (LOESS) to extract these components, making it robust to
outliers and applicable to a wide range of time series data.

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
    .ConfigureModel(new STLDecomposition<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrixVector(x, new Vector<double>(series)))
    .BuildAsync();

var forecast = result.Predict(new Matrix<double>(6, 1));
Console.WriteLine($"STLDecomposition: forecast {forecast.Length} steps.");
```

