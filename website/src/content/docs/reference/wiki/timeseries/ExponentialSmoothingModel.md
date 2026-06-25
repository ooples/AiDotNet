---
title: "ExponentialSmoothingModel<T>"
description: "Represents a model that implements exponential smoothing for time series forecasting."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TimeSeries`

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

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ExponentialSmoothingModel` | Initializes a new instance with default settings. |
| `ExponentialSmoothingModel(ExponentialSmoothingOptions<>)` | Initializes a new instance of the `ExponentialSmoothingModel` class with the specified options. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EsOptions` | Gets the typed options for this model, providing access to ExponentialSmoothing-specific settings. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateMSE(Vector<>,,,)` | Calculates the Mean Squared Error (MSE) for a given set of smoothing parameters. |
| `Clone` | Creates a deep copy of the current exponential smoothing model, including all trained state. |
| `CreateInstance` | Creates a new instance of the exponential smoothing model with the same options. |
| `DeserializeCore(BinaryReader)` | Deserializes the model's core parameters from a binary reader. |
| `EstimateInitialSeasonalFactors(Vector<>)` | Estimates the initial seasonal factors for time series with seasonality. |
| `EstimateInitialValues(Vector<>)` | Estimates the initial values for level, trend, and seasonal components. |
| `EstimateParametersGridSearch(Vector<>)` | Estimates optimal smoothing parameters using a grid search approach. |
| `EvaluateModel(Matrix<>,Vector<>)` | Evaluates the model's performance on test data. |
| `Forecast(Vector<>,Int32)` | Forecasts future values by first replaying smoothing updates over the provided history to align state, then projecting forward for the requested number of steps. |
| `ForecastWithParameters(Vector<>,,,)` | Generates forecasts using specified smoothing parameters. |
| `GetModelMetadata` | Returns metadata about the model, including its type, parameters, and configuration. |
| `Predict(Matrix<>)` | Generates predictions for the given input data. |
| `PredictSingle(Vector<>)` | Predicts a single value based on the input features vector. |
| `Reset` | Resets the model to its initial state. |
| `SaveTrainedState(Vector<>)` | Runs through the training data with the optimized parameters to compute the final level, trend, and seasonal state for use in forecasting. |
| `TrainCore(Matrix<>,Vector<>)` | Implements the core training logic for the exponential smoothing model. |

## Fields

| Field | Summary |
|:-----|:--------|
| `SerializationVersion` | Serializes the model's core parameters to a binary writer. |
| `_alpha` | The smoothing factor for the level component (alpha). |
| `_beta` | The smoothing factor for the trend component (beta). |
| `_gamma` | The smoothing factor for the seasonal component (gamma). |
| `_initialValues` | The initial values for level, trend, and seasonal components. |
| `_trainedLevel` | The level value at the end of training, used as the starting point for forecasting. |
| `_trainedSeasonalFactors` | The seasonal factors at the end of training, used as the starting point for forecasting. |
| `_trainedTrend` | The trend value at the end of training, used as the starting point for forecasting. |
| `_trainingLength` | The number of observations seen during training, used to correctly align the seasonal index when forecasting. |

