---
title: "ARMAModel<T>"
description: "Implements an ARMA (AutoRegressive Moving Average) model for time series forecasting."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TimeSeries`

Implements an ARMA (AutoRegressive Moving Average) model for time series forecasting.

## How It Works

ARMA models combine two components to forecast time series data:

- AR (AutoRegressive): Uses the relationship between an observation and a number of lagged observations
- MA (Moving Average): Uses the relationship between an observation and residual errors from moving average model

For Beginners:
The ARMA model is a popular method for analyzing and forecasting time series data
(data collected over time, like daily temperatures, stock prices, or monthly sales).

Think of ARMA as combining two different approaches:

1. AutoRegressive (AR): This component predicts future values based on past values.

For example, tomorrow's temperature might be related to today's temperature.
If it's hot today, it's likely to be hot tomorrow as well.

2. Moving Average (MA): This component predicts future values based on past prediction errors.

For example, if we consistently underestimate temperature, the MA component
helps adjust our future predictions upward.

The model has two key parameters:

- p: The AR order - how many past values to consider
- q: The MA order - how many past prediction errors to consider

Unlike ARIMA, ARMA doesn't include the differencing (I) component, so it works
best with time series data that is already stationary (doesn't have strong trends).

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
    .ConfigureModel(new ARMAModel<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrixVector(x, new Vector<double>(series)))
    .BuildAsync();

var forecast = result.Predict(new Matrix<double>(6, 1));
Console.WriteLine($"ARMAModel: forecast {forecast.Length} steps.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ARMAModel` | Initializes a new instance with default settings. |
| `ARMAModel(ARMAOptions<>)` | Creates a new ARMA model with the specified options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateGradients(Vector<>,Vector<>)` | Calculates the gradients for adjusting the AR and MA coefficients. |
| `CalculateResiduals(Vector<>)` | Calculates the residuals (prediction errors) for the current model. |
| `CheckConvergence(Vector<>,Vector<>,Vector<>,Vector<>)` | Checks if the training process has converged (reached a stable solution). |
| `Clone` | Creates a deep copy of the current model. |
| `CreateInstance` | Creates a new instance of the ARMA model with the same options. |
| `DeserializeCore(BinaryReader)` | Deserializes the model's state from a binary stream. |
| `EvaluateModel(Matrix<>,Vector<>)` | Evaluates the model's performance on test data. |
| `Forecast(Vector<>,Int32)` | Predicts future values based on a history of time series data. |
| `GetModelMetadata` | Gets metadata about the trained model, including its type, coefficients, and configuration. |
| `Predict(Matrix<>)` | Makes predictions using the trained ARMA model. |
| `Predict(Vector<>,Int32)` | Helper method that predicts a single value at a specific time point. |
| `PredictSingle(Vector<>)` | Predicts a single value based on a single input vector. |
| `SerializeCore(BinaryWriter)` | Serializes the model's state to a binary stream. |
| `ToString` | Returns a string representation of the ARMA model. |
| `TrainCore(Matrix<>,Vector<>)` | Implements the core training algorithm for the ARMA model. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_arCoefficients` | Coefficients for the autoregressive (AR) component of the model. |
| `_arOrder` | The number of past observations to consider (the AR order). |
| `_learningRate` | The step size for gradient descent during training. |
| `_maCoefficients` | Coefficients for the moving average (MA) component of the model. |
| `_maOrder` | The number of past prediction errors to consider (the MA order). |
| `_maxIterations` | The maximum number of iterations for the training algorithm. |
| `_seriesMean` | Mean of the training series, used for centering predictions. |
| `_tolerance` | The convergence threshold for training. |
| `_trainedResiduals` | The residuals (prediction errors) from training. |
| `_trainedSeries` | The time series values used during training. |

