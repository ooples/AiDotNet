---
title: "TimeSeriesRegressionOptions<T>"
description: "Configuration options for time series regression models, which analyze data collected over time to identify patterns and make predictions."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for time series regression models, which analyze data collected over time
to identify patterns and make predictions.

## For Beginners

Time series regression helps you analyze and predict data that changes over time,
like stock prices, weather patterns, or monthly sales figures. Unlike regular regression that just looks for
relationships between variables, time series regression also considers when things happened. It can detect
patterns like "sales always increase in December" or "temperature today is related to temperature yesterday."
This class lets you configure how the model analyzes these time-based patterns.

## How It Works

Time series regression extends traditional regression analysis to account for the temporal nature of data,
where observations are collected sequentially over time. These models can capture trends, seasonal patterns,
and the effects of past values on current and future values.

## Properties

| Property | Summary |
|:-----|:--------|
| `AutocorrelationCorrection` | Gets or sets whether to apply autocorrelation correction to the model. |
| `IncludeTrend` | Gets or sets whether to include a trend component in the model. |
| `LagOrder` | Gets or sets the lag order, which determines how many previous time steps are used as predictors. |
| `LossFunction` | Gets or sets the loss function used for gradient computation and model training. |
| `MaxPredictionAbsValue` | Gets or sets the maximum absolute value allowed for predictions. |
| `ModelType` | Gets or sets the specific type of time series model to use. |
| `SeasonalPeriod` | Gets or sets the seasonal period of the time series data. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_maxTrainingTimeSeconds` | Gets or sets the maximum wall-clock training time in seconds. |

