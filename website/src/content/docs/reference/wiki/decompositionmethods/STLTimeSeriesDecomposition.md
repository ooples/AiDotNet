---
title: "STLTimeSeriesDecomposition<T>"
description: "Implements the Seasonal-Trend decomposition using LOESS (STL) algorithm for time series analysis."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DecompositionMethods.TimeSeriesDecomposition`

Implements the Seasonal-Trend decomposition using LOESS (STL) algorithm for time series analysis.

## For Beginners

Think of this like breaking down your monthly expenses into:

- A trend (are you spending more or less over time?)
- Seasonal patterns (do you spend more during holidays?)
- Unexpected expenses (random costs that don't fit the patterns)

## How It Works

STL decomposition breaks down a time series into three components:
trend (long-term progression), seasonal (recurring patterns), and residual (remaining noise).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `STLTimeSeriesDecomposition(Vector<>,STLDecompositionOptions<>,STLAlgorithmType)` | Initializes a new instance of the STL time series decomposition algorithm. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AdjustForDayOfWeek(,DateTime)` | Adjusts a value based on the day of the week it occurred on. |
| `AdjustForHolidays(,DateTime)` | Adjusts a value if it falls on a holiday. |
| `AdjustForMonthOfYear(,DateTime)` | Adjusts a value based on the month of the year it occurred in. |
| `ApplySeasonalAdjustment(,DateTime)` | Applies seasonal adjustments to a value based on its date. |
| `CalculateSeriesMean` | Calculates the mean (average) of the time series data. |
| `CalculateSeriesStdDev` | Calculates the standard deviation of the time series data. |
| `CreateFastSTLInputMatrix` | Creates an input matrix optimized for the Fast STL algorithm. |
| `Decompose` | Performs the STL decomposition on the time series data. |
| `GetOrCreateDates` | Gets or creates date values for each point in the time series. |
| `ImputeMissingValue(Int32)` | Fills in missing values in the time series. |
| `IsOutlier(,Int32)` | Determines if a value is an outlier using the configured detection method. |
| `IsOutlierIQR()` | Determines if a value is an outlier using the Interquartile Range (IQR) method. |
| `IsOutlierZScore()` | Determines if a value is an outlier using the Z-Score method. |
| `PreprocessValue(,Int32,DateTime)` | Preprocesses a value to handle missing data, outliers, and apply seasonal adjustments. |
| `SmoothOutlier(,Int32)` | Smooths an outlier value by replacing it with the median of nearby values. |

