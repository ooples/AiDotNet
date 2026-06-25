---
title: "TrendFS<T>"
description: "Trend-based feature selection for time series data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.TimeSeries`

Trend-based feature selection for time series data.

## For Beginners

This method looks for features that are consistently
going up or down over time, like a stock price in a bull market. Features without
clear direction (just random fluctuation) score low. It's useful when you care
about the overall trajectory, not just short-term patterns.

## How It Works

Trend-based feature selection identifies features that exhibit significant
temporal trends (upward or downward movement over time). Features with strong
trends may be useful for long-term forecasting or detecting changes.

