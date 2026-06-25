---
title: "ARModel<T>"
description: "Implements an AR (AutoRegressive) model for time series forecasting."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TimeSeries`

Implements an AR (AutoRegressive) model for time series forecasting.

## How It Works

The AR model is a time series forecasting method that uses the relationship between 
an observation and a number of lagged observations to predict future values.

For Beginners:
The AR (AutoRegressive) model is one of the simplest and most intuitive time series
forecasting methods. It's similar to how we naturally predict things in everyday life.

Think of it this way: If you want to guess tomorrow's temperature, you might look at today's
temperature. If it's hot today, it's likely to be hot tomorrow. That's essentially what an
AR model does - it uses past values to predict future values.

For example, if you want to predict today's stock price, an AR model might look at the 
prices from the last few days. If the stock has been trending upward, the model will likely
predict that it continues to rise.

The key parameter is the "AR order" (p), which determines how many past values to consider.
For example:

- AR(1): Only looks at the previous value (yesterday to predict today)
- AR(2): Looks at the previous two values (yesterday and the day before to predict today)
- AR(7): Looks at values from the past week to make predictions

Unlike more complex models like ARMA or ARIMA, the AR model only contains the autoregressive
component and doesn't account for moving average errors or trends that require differencing.

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
    .ConfigureModel(new ARModel<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrixVector(x, new Vector<double>(series)))
    .BuildAsync();

var forecast = result.Predict(new Matrix<double>(6, 1));
Console.WriteLine($"ARModel: forecast {forecast.Length} steps.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ARModel` | Initializes a new instance with default settings. |
| `ARModel(ARModelOptions<>)` | Creates a new AR model with the specified options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateGradients(Vector<>,Vector<>)` | Calculates the gradients for adjusting the AR coefficients. |
| `CalculateResiduals(Vector<>)` | Calculates the residuals (prediction errors) for the current model. |
| `CheckConvergence(Vector<>,Vector<>)` | Checks if the training process has converged (reached a stable solution). |
| `Clone` | Creates a deep copy of the current model. |
| `CreateInstance` | Creates a new instance of the AR model with the same options. |
| `DeserializeCore(BinaryReader)` | Deserializes the model's state from a binary stream. |
| `EvaluateModel(Matrix<>,Vector<>)` | Evaluates the model's performance on test data. |
| `Forecast(Vector<>,Int32)` | Predicts future values based on a history of time series data. |
| `GetModelMetadata` | Gets metadata about the trained model, including its type, coefficients, and configuration. |
| `Predict(Matrix<>)` | Makes predictions using the trained AR model. |
| `Predict(Vector<>,Int32)` | Helper method that predicts a single value at a specific time point. |
| `PredictSingle(Vector<>)` | Predicts a single value based on a single input vector. |
| `Reset` | Resets the model to its untrained state. |
| `SerializeCore(BinaryWriter)` | Serializes the model's state to a binary stream. |
| `ToString` | Returns a string representation of the AR model. |
| `TrainCore(Matrix<>,Vector<>)` | Implements the core training algorithm for the AR model. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_arCoefficients` | Coefficients for the autoregressive (AR) component of the model. |
| `_arOrder` | The number of past observations to consider (the AR order). |
| `_learningRate` | The step size for gradient descent during training. |
| `_maxIterations` | The maximum number of iterations for the training algorithm. |
| `_tolerance` | The convergence threshold for training. |
| `_trainedSeries` | The time series values from training, used for in-sample predictions via Predict(Matrix). |

