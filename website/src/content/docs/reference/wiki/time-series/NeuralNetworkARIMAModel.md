---
title: "NeuralNetworkARIMAModel"
description: "Represents a Neural Network ARIMA (Autoregressive Integrated Moving Average) model for time series forecasting."
section: "Reference"
---

_Time-Series Models_

Represents a Neural Network ARIMA (Autoregressive Integrated Moving Average) model for time series forecasting.

## For Beginners

This model is like a super-powered crystal ball for predicting future values in a sequence of data. Imagine you're trying to predict tomorrow's temperature: - The ARIMA part looks at recent temperatures and how they've been changing. - The Neural Network part can spot complex patterns, like how weekends or holidays might affect temperature. By combining these two approaches, this model can make more accurate predictions than either method alone. It's especially useful for data that changes over time, like stock prices, weather patterns, or sales figures.

## How It Works

This class combines traditional ARIMA modeling with neural networks to create a hybrid model for time series forecasting. It incorporates both linear (ARIMA) and non-linear (neural network) components to capture complex patterns in the data.

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
    .ConfigureModel(new NeuralNetworkARIMAModel<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrixVector(x, new Vector<double>(series)))
    .BuildAsync();

var forecast = result.Predict(new Matrix<double>(6, 1));
Console.WriteLine($"NeuralNetworkARIMAModel: forecast {forecast.Length} steps.");
```

