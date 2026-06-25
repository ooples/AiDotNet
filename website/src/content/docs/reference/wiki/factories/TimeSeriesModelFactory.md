---
title: "TimeSeriesModelFactory<T, TInput, TOutput>"
description: "A factory class that creates time series models for forecasting and analysis."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Factories`

A factory class that creates time series models for forecasting and analysis.

## For Beginners

Time series models are specialized algorithms that analyze data collected over time 
(like daily temperatures, monthly sales, or yearly population) to identify patterns and make predictions 
about future values.

## How It Works

This factory helps you create different types of time series models without needing to know their 
internal implementation details. Think of it like ordering a specific tool from a catalog - you just 
specify what you need, and the factory provides it.

## Methods

| Method | Summary |
|:-----|:--------|
| `ConvertToModelSpecificOptions(TimeSeriesModelType,TimeSeriesRegressionOptions<>)` | Converts base TimeSeriesRegressionOptions to the appropriate model-specific options type. |

