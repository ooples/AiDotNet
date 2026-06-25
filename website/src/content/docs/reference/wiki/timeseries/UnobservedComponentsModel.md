---
title: "UnobservedComponentsModel<T, TInput, TOutput>"
description: "Implements an Unobserved Components Model (UCM) for time series decomposition and forecasting."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TimeSeries`

Implements an Unobserved Components Model (UCM) for time series decomposition and forecasting.

## How It Works

The Unobserved Components Model decomposes a time series into several distinct components:
trend, seasonal, cycle, and irregular components. It uses state-space modeling and Kalman filtering
to estimate these components, which can then be used for forecasting or understanding the
underlying patterns in the data.

For Beginners:
An Unobserved Components Model is like having X-ray vision for your time series data.
It helps you see the hidden patterns that make up your data by breaking it down into
several meaningful parts:

1. Trend Component: The long-term direction of your data. Is it generally going up,

down, or staying level over time? This is like the "big picture" movement.

2. Seasonal Component: Regular patterns that repeat at fixed intervals, such as

daily, weekly, monthly, or yearly cycles. For example, retail sales might
spike every December for holiday shopping.

3. Cycle Component: Longer-term ups and downs that don't have a fixed period, often

related to business or economic cycles. Unlike seasonal patterns, these aren't tied
to the calendar and can vary in length and intensity.

4. Irregular Component: The random "noise" or unexpected fluctuations that don't fit

into the other components. This captures events like unusual weather, one-time promotions,
or other unpredictable factors.

The model uses a mathematical technique called Kalman filtering (a bit like a sophisticated
version of moving averages) to separate these components from your data. Once separated,
you can examine each component individually to better understand what's driving your time series,
or recombine them to make forecasts.

This approach is particularly valuable because it:

- Helps you understand the "why" behind your data's behavior
- Allows you to forecast each component separately, improving accuracy
- Makes it easier to spot unusual patterns or anomalies
- Provides insights that simpler models might miss

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `UnobservedComponentsModel(UnobservedComponentsOptions<,,>)` | Creates a new Unobserved Components Model with the specified options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BandPassFilter(Vector<>,Int32,Int32)` | Applies a band-pass filter to isolate specific frequency components. |
| `CreateInstance` | Creates a new instance of the UnobservedComponentsModel class. |
| `DeserializeCore(BinaryReader)` | Deserializes the model's state from a binary stream. |
| `EvaluateModel(Matrix<>,Vector<>)` | Evaluates the model's performance on test data. |
| `Forecast(Int32,Int32)` | Forecasts multiple steps into the future using the decomposed components. |
| `ForecastCycle(Int32,Int32)` | Forecasts the cycle component for future time points. |
| `ForecastIrregular(Int32)` | Generates small random variations for the irregular component forecast. |
| `ForecastSeasonal(Int32,Int32)` | Forecasts the seasonal component for future time points. |
| `ForecastTrend(Int32,Int32)` | Forecasts the trend component for future time points. |
| `GetComponents` | Gets the decomposed components of the time series. |
| `GetModelMetadata` | Gets metadata about the model, including its type, components, and configuration. |
| `HasConverged` | Checks if the model has converged based on changes in the trend component. |
| `HodrickPrescottFilter(Vector<>,Double)` | Applies a Hodrick-Prescott filter to separate trend and cyclical components. |
| `InitializeComponents(Vector<>)` | Initializes the trend, seasonal, cycle, and irregular components. |
| `InitializeCycle(Vector<>,Vector<>,Vector<>)` | Initializes the cycle component using filtering techniques. |
| `InitializeKalmanParameters` | Initializes the parameters for the Kalman filter. |
| `InitializeSeasonal(Vector<>,Vector<>)` | Initializes the seasonal component using averaging across periods. |
| `KalmanFilter(Vector<>)` | Applies the Kalman filter to estimate state at each time point. |
| `KalmanSmoother(Vector<>)` | Applies the Kalman smoother to refine state estimates using future information. |
| `MovingAverage(Vector<>,Int32)` | Calculates a simple moving average of the data. |
| `OptimizeParameters(Matrix<>,Vector<>)` | Optimizes the model parameters to better fit the data. |
| `Predict(Matrix<>)` | Makes predictions using the trained model. |
| `PredictSingle(Vector<>)` | Generates a prediction for a single input vector. |
| `Reset` | Resets the model to its untrained state. |
| `SerializeCore(BinaryWriter)` | Serializes the model's state to a binary stream. |
| `TrainCore(Matrix<>,Vector<>)` | Implements the model-specific training logic for the Unobserved Components Model. |
| `UpdateComponentsFromSmoothedState(List<Vector<>>)` | Updates the model components based on the smoothed state estimates. |
| `UpdateModelParameters(Vector<>)` | Updates the model parameters based on optimized values. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_convergenceThreshold` | The threshold for determining when the model has converged. |
| `_cycle` | The estimated cycle component of the time series. |
| `_fft` | Fast Fourier Transform utility for frequency analysis. |
| `_filteredCovariance` | Collection of filtered state covariance matrices from the Kalman filter. |
| `_filteredState` | Collection of filtered state vectors from the Kalman filter. |
| `_irregular` | The estimated irregular component of the time series. |
| `_observationModel` | The observation model matrix for the Kalman filter. |
| `_observationNoise` | The observation noise variance for the Kalman filter. |
| `_previousTrend` | The previous iteration's trend estimates, used to check for convergence. |
| `_processNoise` | The process noise covariance matrix for the Kalman filter. |
| `_seasonal` | The estimated seasonal component of the time series. |
| `_state` | The current state vector for the Kalman filter. |
| `_stateCovariance` | The current state covariance matrix for the Kalman filter. |
| `_stateTransition` | The state transition matrix for the Kalman filter. |
| `_trend` | The estimated trend component of the time series. |
| `_ucOptions` | Configuration options for the Unobserved Components Model. |
| `_y` | The original time series data. |

