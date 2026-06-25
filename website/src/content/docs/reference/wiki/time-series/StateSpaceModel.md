---
title: "StateSpaceModel"
description: "Implements a State Space Model for time series analysis and forecasting."
section: "Reference"
---

_Time-Series Models_

Implements a State Space Model for time series analysis and forecasting.

## For Beginners

A State Space Model is like tracking the position of a moving object when you can only see its shadow. The actual position (state) is hidden, but you can observe its effects (the shadow). For example, if you're tracking the economy, you might not directly observe the "true state" of the economy, but you can see indicators like GDP, unemployment rates, etc. The State Space Model helps infer the hidden state from these observations and predict future values. The model has two main components: 1. A transition equation that describes how the hidden state evolves over time 2. An observation equation that relates the hidden state to what we actually observe This implementation uses the Kalman filter and smoother algorithms to estimate the hidden states and learn the model parameters from data.

## How It Works

State Space Models represent time series data as a system with hidden states that evolve over time according to probabilistic rules. They are powerful tools for modeling complex dynamic systems and can handle missing data, multiple variables, and non-stationary patterns.

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
    .ConfigureModel(new StateSpaceModel<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrixVector(x, new Vector<double>(series)))
    .BuildAsync();

var forecast = result.Predict(new Matrix<double>(6, 1));
Console.WriteLine($"StateSpaceModel: forecast {forecast.Length} steps.");
```

