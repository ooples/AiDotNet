---
title: "ExponentialSmoothingOptions<T>"
description: "Configuration options for Exponential Smoothing, a time series forecasting method that gives exponentially decreasing weights to older observations."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Exponential Smoothing, a time series forecasting method that gives
exponentially decreasing weights to older observations.

## For Beginners

Exponential smoothing is like a weighted average that gives more importance
to recent data points and less importance to older ones. Imagine predicting tomorrow's temperature:
you might care more about today's temperature than what happened two weeks ago. This method works
similarly, gradually "forgetting" older data while focusing on newer trends. It's particularly good
for data that changes over time and where recent observations are more relevant for prediction.
This class lets you configure how quickly the model "forgets" old data and how it handles trends
and seasonal patterns.

## How It Works

Exponential smoothing is a popular forecasting method for time series data that works by applying
weighted averages where the weights decrease exponentially as observations get older. This means
recent observations have more influence on forecasts than older observations.

## Properties

| Property | Summary |
|:-----|:--------|
| `GridSearchStep` | Gets or sets the step size for grid search when optimizing smoothing parameters. |
| `InitialAlpha` | Gets or sets the initial smoothing factor (alpha) that controls the weight given to recent observations. |
| `InitialBeta` | Gets or sets the initial trend smoothing factor (beta) that controls the weight given to the trend component. |
| `InitialGamma` | Gets or sets the initial seasonal smoothing factor (gamma) that controls the weight given to the seasonal component. |
| `UseSeasonal` | Gets or sets whether to include a seasonal component in the exponential smoothing model. |
| `UseTrend` | Gets or sets whether to include a trend component in the exponential smoothing model. |

