---
title: "LagFeatureSelector<T>"
description: "Lag-based Feature Selection for Time Series."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.TimeSeries`

Lag-based Feature Selection for Time Series.

## For Beginners

In time series, past values often predict future
values. This selector finds features whose past values (lags) are strongly
correlated with the target, helping you identify which features are most
useful for forecasting.

## How It Works

Selects features based on their lagged correlations with the target,
identifying features that have predictive power at different time lags.

