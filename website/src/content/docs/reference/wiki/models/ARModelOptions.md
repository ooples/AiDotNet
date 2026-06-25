---
title: "ARModelOptions<T>"
description: "Configuration options for the AR (AutoRegressive) time series forecasting model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the AR (AutoRegressive) time series forecasting model.

## For Beginners

An AutoRegressive (AR) model is one of the simplest methods for predicting future values 
in a time series (data collected over time, like daily temperatures or monthly sales). It works on the principle that 
future values can be predicted by looking at past values. Think of it like predicting tomorrow's weather primarily based 
on what the weather has been like for the past few days. Unlike more complex models like ARMA or ARIMA, the AR model 
focuses solely on the relationship between current values and previous values, without considering prediction errors 
or trends.

## How It Works

The AR model is a statistical approach for analyzing and forecasting time series data. It predicts future values
based on a linear combination of previous values in the time series.

## Properties

| Property | Summary |
|:-----|:--------|
| `AROrder` | Gets or sets the order of the AutoRegressive (AR) component. |
| `LearningRate` | Gets or sets the learning rate for the optimization algorithm. |
| `MaxIterations` | Gets or sets the maximum number of iterations for the optimization algorithm. |
| `Tolerance` | Gets or sets the convergence tolerance for the optimization algorithm. |

