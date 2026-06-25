---
title: "ARIMAModel<T>"
description: "Implements an ARIMA (AutoRegressive Integrated Moving Average) model for time series forecasting."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TimeSeries`

Implements an ARIMA (AutoRegressive Integrated Moving Average) model for time series forecasting.

## For Beginners

ARIMA is a popular technique for analyzing and forecasting time series data (data collected over time,
like stock prices, temperature readings, or monthly sales figures).

## How It Works

ARIMA models are widely used for time series forecasting. The model combines three components:

- AR (AutoRegressive): Uses the dependent relationship between an observation and a number of lagged observations
- I (Integrated): Uses differencing of observations to make the time series stationary
- MA (Moving Average): Uses the dependency between an observation and residual errors from a moving average model

Think of ARIMA as combining three different approaches:

The model has three key parameters (p, d, q):

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
    .ConfigureModel(new ARIMAModel<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrixVector(x, new Vector<double>(series)))
    .BuildAsync();

var forecast = result.Predict(new Matrix<double>(6, 1));
Console.WriteLine($"ARIMAModel: forecast {forecast.Length} steps.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ARIMAModel(ARIMAOptions<>)` | Creates a new ARIMA model with the specified options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeAnomalyScores(Vector<>)` | Computes anomaly scores for each point in a time series. |
| `ComputeAnomalyThreshold(Vector<>)` | Computes the anomaly detection threshold from training residuals. |
| `ComputeIntegrationTailValues(Vector<>,Int32)` | Computes the tail value at each integration (differencing) level from the history, needed for undifferencing forecasts back to the original scale. |
| `CreateInstance` | Creates a new instance of the ARIMA model with the same options. |
| `DeserializeCore(BinaryReader)` | Deserializes the model's state from a binary stream. |
| `DetectAnomalies(Vector<>)` | Detects anomalies in a time series by comparing predictions to actual values. |
| `DetectAnomaliesDetailed(Vector<>)` | Detects anomalies and returns detailed information about each detected anomaly. |
| `EstimateConstant(Vector<>,Vector<>,Vector<>)` | Estimates the constant term for the ARIMA model. |
| `EvaluateModel(Matrix<>,Vector<>)` | Evaluates the model's performance on test data. |
| `Forecast(Vector<>,Int32)` | Forecasts future values using the trained ARIMA model, properly handling differencing. |
| `GetAnomalyThreshold` | Gets the current anomaly detection threshold. |
| `GetModelMetadata` | Gets metadata about the model, including its type, parameters, and configuration. |
| `Predict(Matrix<>)` | Makes predictions using the trained ARIMA model. |
| `PredictSingle(Vector<>)` | Predicts a single value based on the input vector. |
| `SerializeCore(BinaryWriter)` | Serializes the model's state to a binary stream. |
| `SetAnomalyThreshold()` | Sets a custom anomaly detection threshold. |
| `TrainCore(Matrix<>,Vector<>)` | Core implementation of the training logic for the ARIMA model. |
| `UndifferenceForecasts(Vector<>,Vector<>,Int32,Int32)` | Undifferences forecasted values back to the original scale by reversing the differencing that was applied during training. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_anomalyThreshold` | The anomaly detection threshold computed during training. |
| `_arCoefficients` | Coefficients for the autoregressive (AR) component of the model. |
| `_arimaOptions` | Options specific to the ARIMA model, including p, d, and q parameters. |
| `_constant` | The constant term in the ARIMA equation. |
| `_lastTrainDiffValues` | Stored differenced training series values (last P values) for initializing Predict(Matrix). |
| `_lastTrainResiduals` | Stored AR residuals from training (last Q values) for initializing Predict(Matrix). |
| `_maCoefficients` | Coefficients for the moving average (MA) component of the model. |
| `_residualMean` | The mean of residuals computed during training. |
| `_residualStdDev` | The standard deviation of residuals computed during training. |
| `_trainingSeries` | Stored original training series for in-sample prediction and Forecast initialization. |

