---
title: "ProphetModel<T, TInput, TOutput>"
description: "Represents a Prophet model for time series forecasting."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TimeSeries`

Represents a Prophet model for time series forecasting.

## For Beginners

Think of the Prophet model as a smart crystal ball for predicting future values 
in a series of data points over time. It's like predicting weather, but for any kind of data that changes 
over time, such as sales, website traffic, or stock prices. The model looks at past patterns, including 
seasonal changes (like how sales might increase during holidays) and overall trends, to make educated 
guesses about future values.

## How It Works

The Prophet model is a procedure for forecasting time series data based on an additive model 
where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ProphetModel(ProphetOptions<,,>)` | Initializes a new instance of the `ProphetModel<T>` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsOptimized` | Gets whether parameter optimization succeeded during the most recent training run. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeAnomalyScores(Matrix<>,Vector<>)` | Computes anomaly scores for each point in a time series. |
| `ComputeAnomalyThresholdFromTraining(Matrix<>,Vector<>)` | Computes the anomaly detection threshold from training data residuals. |
| `ComputeAverageChangepointEffect` | Computes the average changepoint effect for JIT approximation. |
| `ComputeAverageHolidayEffect` | Computes the average holiday effect for JIT approximation. |
| `ComputeAverageSeasonalEffect` | Computes the average seasonal effect for JIT approximation. |
| `CreateInstance` | Creates a new instance of the Prophet model with the same options. |
| `DeserializeCore(BinaryReader)` | Deserializes the core components of the Prophet model. |
| `DetectAnomalies(Matrix<>,Vector<>)` | Detects anomalies in a time series by comparing predictions to actual values. |
| `EstimateInitialChangepoint(Vector<>)` | Estimates the initial changepoint value based on the input data. |
| `EstimateInitialTrend(Vector<>)` | Estimates the initial trend of the time series data. |
| `EvaluateModel(Matrix<>,Vector<>)` | Evaluates the performance of the Prophet model on a test dataset. |
| `GetAnomalyThreshold` | Gets the current anomaly detection threshold. |
| `GetChangepointEffect(Vector<>)` | Calculates the changepoint effect of the time series for a given input vector. |
| `GetCurrentState` | Retrieves the current state of the model as a vector. |
| `GetHolidayComponent(Vector<>)` | Calculates the holiday component of the time series for a given input vector. |
| `GetModelMetadata` | Gets metadata about the model, including its type, parameters, and configuration. |
| `GetRegressorEffect(Vector<>)` | Calculates the regressor effect of the time series for a given input vector. |
| `GetSeasonalComponent(Vector<>)` | Calculates the seasonal component of the time series for a given input vector. |
| `GetStateSize` | Calculates the total size of the model's state vector. |
| `InitializeComponents(Matrix<>,Vector<>)` | Initializes the components of the Prophet model based on the input data. |
| `InitializeHolidayComponents(Matrix<>,Vector<>)` | Initializes the holiday components of the model. |
| `InitializeRegressors(Matrix<>,Vector<>)` | Initializes the regressor components of the model. |
| `InitializeSeasonalComponents(Matrix<>,Vector<>)` | Initializes the seasonal components of the Prophet model. |
| `OptimizeParameters(Matrix<>,Vector<>)` | Optimizes the model parameters using the specified or default optimizer. |
| `Predict(Matrix<>)` | Predicts output values for the given input matrix. |
| `PredictSingle(Vector<>)` | Predicts a single value based on the input vector. |
| `PredictSingleInternal(Vector<>)` | Predicts a single output value for the given input vector. |
| `PredictWithIntervals(Matrix<>)` | Computes prediction intervals for future forecasts. |
| `SerializeCore(BinaryWriter)` | Serializes the core components of the Prophet model. |
| `SetAnomalyThreshold()` | Sets a custom anomaly detection threshold. |
| `SimpleLinearRegression(Vector<>,Vector<>)` | Performs a simple linear regression to find the relationship between two variables. |
| `SimpleMultipleRegression(Matrix<>,Vector<>)` | Performs simple multiple linear regression. |
| `TrainCore(Matrix<>,Vector<>)` | Core implementation of the training logic for the Prophet model. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_anomalyThreshold` | The anomaly detection threshold computed during training. |
| `_changepoint` | Represents the coefficient for the changepoint component. |
| `_holidayComponents` | Stores the effect of each holiday on the time series. |
| `_prophetOptions` | Stores the configuration options for the Prophet model. |
| `_regressors` | Stores the coefficients for additional regressor variables. |
| `_residualMean` | The mean of residuals computed during training. |
| `_residualStdDev` | The standard deviation of residuals computed during training. |
| `_seasonalComponents` | Stores the coefficients for all seasonal components. |
| `_trend` | Represents the overall trend component of the time series. |

