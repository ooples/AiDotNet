---
title: "LaggedCorrelationSelector<T>"
description: "Lagged correlation selector for time series feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.TimeSeries`

Lagged correlation selector for time series feature selection.

## For Beginners

This finds features whose current values
predict future values of the target. A feature might correlate with
the target 3 time steps later, making it useful for forecasting.

## How It Works

Computes cross-correlation between features and target at various lags
to find features that predict future target values.

