---
title: "AutocorrelationFS<T>"
description: "Autocorrelation-based feature selection for time series data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.TimeSeries`

Autocorrelation-based feature selection for time series data.

## For Beginners

Autocorrelation asks: "How similar is today's value
to yesterday's, the day before, etc.?" Features with high autocorrelation have
patterns that repeat or persist over time, making them potentially valuable for
predicting future values.

## How It Works

Autocorrelation measures how a feature correlates with itself at different time lags.
Features with strong autocorrelation patterns often contain predictable structure
that can be useful for time series forecasting.

