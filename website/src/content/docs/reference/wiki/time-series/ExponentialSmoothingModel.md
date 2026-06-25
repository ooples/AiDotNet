---
title: "ExponentialSmoothingModel"
description: "Represents a model that implements exponential smoothing for time series forecasting."
section: "Reference"
---

_Time-Series Models_

Represents a model that implements exponential smoothing for time series forecasting.

## For Beginners

Exponential smoothing helps predict future values based on past data.

Think of it like predicting tomorrow's weather:

- Recent weather (yesterday, today) is more important than weather from weeks ago
- You can identify trends (getting warmer over time)
- You can account for seasons (summer is usually warmer than winter)

For example, if you're forecasting daily sales:

- Simple smoothing: Uses a weighted average of past values, giving more weight to recent sales
- Double smoothing: Also captures if sales are trending up or down
- Triple smoothing: Adds seasonal patterns (e.g., higher sales on weekends)

Exponential smoothing is called "exponential" because the weight given to older data
decreases exponentially as the data gets older.

## How It Works

Exponential smoothing is a time series forecasting method that assigns exponentially decreasing weights
to past observations, giving more importance to recent data while still considering older observations.
This model supports simple, double (with trend), and triple (with trend and seasonality) exponential smoothing.

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
    .ConfigureModel(new ExponentialSmoothingModel<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrixVector(x, new Vector<double>(series)))
    .BuildAsync();

var forecast = result.Predict(new Matrix<double>(6, 1));
Console.WriteLine($"ExponentialSmoothingModel: forecast {forecast.Length} steps.");
```

