---
title: "ARMAOptions<T>"
description: "Configuration options for the ARMA (AutoRegressive Moving Average) time series forecasting model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the ARMA (AutoRegressive Moving Average) time series forecasting model.

## For Beginners

ARMA is a method for predicting future values in a time series (data collected 
over time, like daily stock prices or monthly sales figures). It works by combining two techniques: 
looking at how past values influence future ones (AutoRegressive) and accounting for the impact of past 
prediction errors (Moving Average). Think of it like predicting tomorrow's weather by considering both 
today's weather (AR component) and how accurate previous forecasts have been (MA component). Unlike ARIMA, 
ARMA assumes your data doesn't have strong upward or downward trends that need to be removed first.

## How It Works

ARMA is a statistical model used for analyzing and forecasting time series data. It combines two components:
AutoRegressive (AR) and Moving Average (MA) to model stationary time series data.

## Properties

| Property | Summary |
|:-----|:--------|
| `AROrder` | Gets or sets the order of the AutoRegressive (AR) component. |
| `LearningRate` | Gets or sets the learning rate for the optimization algorithm. |
| `MAOrder` | Gets or sets the order of the Moving Average (MA) component. |
| `MaxIterations` | Gets or sets the maximum number of iterations for the optimization algorithm. |
| `Tolerance` | Gets or sets the convergence tolerance for the optimization algorithm. |

