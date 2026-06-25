---
title: "SeasonalityFS<T>"
description: "Seasonality-based feature selection for time series data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.TimeSeries`

Seasonality-based feature selection for time series data.

## For Beginners

Many real-world patterns repeat: daily traffic peaks,
monthly sales cycles, yearly temperature changes. This method finds features that
have such repeating patterns. It uses frequency analysis (like detecting musical
notes in sound) to identify periodic behavior in data.

## How It Works

Seasonality feature selection identifies features with periodic patterns at specified
frequencies. It uses spectral analysis to detect features with strong seasonal
components, which can be valuable for forecasting cyclical phenomena.

