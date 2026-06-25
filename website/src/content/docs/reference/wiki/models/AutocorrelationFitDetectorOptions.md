---
title: "AutocorrelationFitDetectorOptions"
description: "Configuration options for detecting autocorrelation in time series data and regression residuals."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for detecting autocorrelation in time series data and regression residuals.

## For Beginners

Autocorrelation is a pattern where data points in a time series are related to 
their own past values. Think of it like weather patterns - if it's been raining for several days, there's a 
higher chance it will rain tomorrow (positive autocorrelation). This class helps determine if your data has 
such patterns by setting thresholds for the Durbin-Watson test, which is like a thermometer for measuring 
autocorrelation. Values around 2 suggest no autocorrelation, values closer to 0 suggest positive autocorrelation 
(each value tends to be similar to previous values), and values closer to 4 suggest negative autocorrelation 
(each value tends to be opposite to previous values). Understanding autocorrelation helps choose the right 
prediction model for your data.

## How It Works

This class provides threshold values used to interpret the Durbin-Watson statistic, which measures
autocorrelation in the residuals (errors) of regression and time series models. The Durbin-Watson
statistic typically ranges from 0 to 4, with different values indicating different types of autocorrelation.

## Properties

| Property | Summary |
|:-----|:--------|
| `NoAutocorrelationLowerBound` | Gets or sets the lower bound of the range indicating no autocorrelation. |
| `NoAutocorrelationUpperBound` | Gets or sets the upper bound of the range indicating no autocorrelation. |
| `StrongNegativeAutocorrelationThreshold` | Gets or sets the threshold for detecting strong negative autocorrelation. |
| `StrongPositiveAutocorrelationThreshold` | Gets or sets the threshold for detecting strong positive autocorrelation. |

