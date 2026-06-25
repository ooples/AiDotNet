---
title: "MultiplicativeAlgorithmType"
description: "Represents different multiplicative algorithm types for time series analysis and forecasting."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums.AlgorithmTypes`

Represents different multiplicative algorithm types for time series analysis and forecasting.

## For Beginners

Multiplicative algorithms are special methods used when analyzing data that changes over time 
(time series data), especially when the pattern of change depends on the current value.

Think about the difference between:

1. Adding $100 to your savings each month (additive growth)
2. Growing your savings by 5% each month (multiplicative growth)

With multiplicative patterns, the changes get larger as the base value gets larger. For example, 5% of $1000 
is $50, but 5% of $10,000 is $500 - the same percentage creates bigger absolute changes as the value grows.

Multiplicative algorithms are especially useful for:

1. Financial data (stock prices, sales figures)
2. Population growth
3. Seasonal patterns that grow or shrink proportionally to the overall trend
4. Any data where percentage changes are more important than absolute changes

In contrast to additive methods (which use addition and subtraction), multiplicative methods use 
multiplication and division to model changes. They often work with data on a logarithmic scale or 
with ratios rather than differences.

This enum specifies which specific multiplicative algorithm to use for analyzing or forecasting time series data.

## Fields

| Field | Summary |
|:-----|:--------|
| `GeometricMovingAverage` | Uses a Geometric Moving Average to analyze and forecast time series data. |
| `LogTransformedSTL` | Uses a log-transformed Seasonal and Trend decomposition using Loess (STL) to analyze time series data. |
| `MultiplicativeExponentialSmoothing` | Uses Multiplicative Exponential Smoothing to analyze and forecast time series data. |

