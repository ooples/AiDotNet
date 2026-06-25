---
title: "BayesianStructuralTimeSeriesModel<T>"
description: "Implements a Bayesian Structural Time Series model for flexible time series forecasting."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TimeSeries`

Implements a Bayesian Structural Time Series model for flexible time series forecasting.

## How It Works

The Bayesian Structural Time Series (BSTS) model is a powerful and flexible approach for analyzing
and forecasting time series data. It decomposes a time series into interpretable components
including level, trend, seasonality, and regression effects, using Bayesian methods to handle
uncertainty and combine information from different sources.

For Beginners:
A Bayesian Structural Time Series model is like having a flexible toolkit for forecasting time series data.
Unlike simpler models like AR or ARIMA that use fixed patterns, BSTS breaks down your data into
meaningful components:

1. Level: The current "baseline" value of your series
2. Trend: Whether your data is generally increasing or decreasing over time
3. Seasonal patterns: Regular cycles in your data (daily, weekly, yearly, etc.)
4. Effects of external factors: How other variables influence your data

The "Bayesian" part means the model handles uncertainty well. Instead of making single point predictions,
it gives you a range of possible outcomes with probabilities attached. It uses something called a
"Kalman filter" to continually update its understanding as new data arrives.

BSTS models are especially powerful because they:

- Can handle missing data
- Allow you to incorporate external information
- Provide predictions with uncertainty ranges
- Let you see which components are driving your forecast

This makes them ideal for analyzing complex time series where you want to understand
what's driving changes in addition to making predictions.

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
    .ConfigureModel(new BayesianStructuralTimeSeriesModel<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrixVector(x, new Vector<double>(series)))
    .BuildAsync();

var forecast = result.Predict(new Matrix<double>(6, 1));
Console.WriteLine($"BayesianStructuralTimeSeriesModel: forecast {forecast.Length} steps.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BayesianStructuralTimeSeriesModel(BayesianStructuralTimeSeriesOptions<>)` | Creates a new Bayesian Structural Time Series model with the specified options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateInnovation(,Vector<>)` | Calculates the innovation (difference between observation and prediction). |
| `CalculateKalmanGain(Matrix<>)` | Calculates the Kalman gain, which determines how much to adjust the state based on new observations. |
| `CalculatePrediction(Vector<>)` | Calculates a prediction from a state vector. |
| `CheckConvergence(,)` | Checks if the parameter estimation has converged. |
| `Clone` | Creates a deep copy of the current model. |
| `CreateInstance` | Creates a new instance of the BSTS model with the same options. |
| `CreateObservationVector` | Creates the observation vector for the state-space model. |
| `CreateProcessNoiseMatrix` | Creates the process noise matrix for the state-space model. |
| `CreateTransitionMatrix` | Creates the transition matrix for the state-space model. |
| `DeserializeCore(BinaryReader)` | Deserializes the model's state from a binary stream. |
| `EstimateParameters(Matrix<>,Vector<>,Matrix<>)` | Estimates optimal parameters for the model using an EM-like algorithm. |
| `EvaluateModel(Matrix<>,Vector<>)` | Evaluates the model's performance on test data. |
| `Forecast(Vector<>,Int32,Matrix<>)` | Forecasts future values based on a history of time series data and optional exogenous variables. |
| `GetCurrentState` | Gets the current state vector for all model components. |
| `GetModelMetadata` | Gets metadata about the trained model, including its type, components, and configuration. |
| `GetStateSize` | Gets the total size of the state vector. |
| `InitializeRegressionCoefficients(Matrix<>,Vector<>)` | Initializes the regression coefficients using Ordinary Least Squares or Ridge Regression. |
| `PerformBackwardSmoothing(Matrix<>)` | Performs backward smoothing to improve state estimates using future information. |
| `Predict(Matrix<>)` | Makes predictions using the trained BSTS model. |
| `PredictCovariance` | Predicts the covariance matrix for the next state. |
| `PredictSingle(Vector<>)` | Predicts a single value based on a single input vector of external regressors. |
| `PredictState(Vector<>)` | Predicts the next state of the time series using the current model. |
| `Reset` | Resets the model to its untrained state. |
| `RunKalmanFilterAndSmoother(Matrix<>,Vector<>)` | Runs a single iteration of Kalman filter and smoother. |
| `SerializeCore(BinaryWriter)` | Serializes the model's state to a binary stream. |
| `TrainCore(Matrix<>,Vector<>)` | Implements the core training algorithm for the Bayesian Structural Time Series model. |
| `UpdateCovariance(Matrix<>,Vector<>)` | Updates the state covariance matrix based on the Kalman gain. |
| `UpdateModelComponentsFromSmoothedStates(Matrix<>)` | Updates model components from the smoothed states. |
| `UpdateModelParameters(Matrix<>,Vector<>,Matrix<>)` | Updates model parameters based on smoothed states. |
| `UpdateState(Vector<>,Vector<>,)` | Updates the state vector based on the Kalman gain and innovation. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_bayesianOptions` | Configuration options for the Bayesian Structural Time Series model. |
| `_level` | The current level (baseline value) of the time series. |
| `_observationVariance` | The estimated variance (uncertainty) in observations. |
| `_regression` | Coefficients for the regression component (impact of external variables). |
| `_seasonalComponents` | The seasonal components of the time series, representing cyclical patterns. |
| `_stateCovariance` | The uncertainty in the state estimates, represented as a covariance matrix. |
| `_trainingSeries` | Stored training series for in-sample predictions. |
| `_trend` | The current trend (rate of change) of the time series. |

