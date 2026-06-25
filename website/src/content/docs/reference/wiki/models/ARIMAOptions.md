---
title: "ARIMAOptions<T>"
description: "Configuration options for the ARIMA (AutoRegressive Integrated Moving Average) time series forecasting model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the ARIMA (AutoRegressive Integrated Moving Average) time series forecasting model.

## For Beginners

ARIMA is a popular method for predicting future values in a time series (data collected 
over time, like daily temperatures or monthly sales). It works by combining three techniques: looking at how past values 
influence future ones (AutoRegressive), removing trends by taking differences between consecutive values (Integrated), 
and accounting for the impact of past prediction errors (Moving Average). Think of it like predicting tomorrow's weather 
by considering today's weather, the recent trend of warming or cooling, and how accurate previous forecasts have been.

## How It Works

ARIMA is a statistical model used for analyzing and forecasting time series data. It combines three components:
AutoRegressive (AR), Integrated (I), and Moving Average (MA) to model time series data that exhibits non-stationarity.

## Properties

| Property | Summary |
|:-----|:--------|
| `AnomalyThresholdSigma` | Gets or sets the number of standard deviations from the mean residual to use as the anomaly threshold. |
| `D` | Gets or sets the order of differencing (Integration). |
| `EnableAnomalyDetection` | Gets or sets a value indicating whether to enable anomaly detection during and after training. |
| `FitIntercept` | Gets or sets whether to include an intercept (constant term) in the model. |
| `LearningRate` | Gets or sets the learning rate for the optimization algorithm. |
| `MaxIterations` | Gets or sets the maximum number of iterations for the optimization algorithm. |
| `P` | Gets or sets the order of the AutoRegressive (AR) component. |
| `Q` | Gets or sets the order of the Moving Average (MA) component. |
| `SeasonalD` | Gets or sets the seasonal differencing order for seasonal ARIMA models. |
| `SeasonalP` | Gets or sets the seasonal AutoRegressive order for seasonal ARIMA models. |
| `SeasonalQ` | Gets or sets the seasonal Moving Average order for seasonal ARIMA models. |
| `Tolerance` | Gets or sets the convergence tolerance for the optimization algorithm. |

