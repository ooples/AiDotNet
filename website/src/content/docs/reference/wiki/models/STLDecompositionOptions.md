---
title: "STLDecompositionOptions<T>"
description: "Configuration options for Seasonal-Trend-Loess (STL) decomposition, a versatile method for decomposing time series into seasonal, trend, and residual components."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Seasonal-Trend-Loess (STL) decomposition, a versatile method
for decomposing time series into seasonal, trend, and residual components.

## For Beginners

STL decomposition helps break down your time series into meaningful components.

When analyzing time series data:

- It's often useful to separate different patterns in the data
- STL decomposition splits your data into three parts:
1. Seasonal component: Repeating patterns (daily, weekly, monthly, etc.)
2. Trend component: Long-term direction (increasing, decreasing, etc.)
3. Residual component: What remains after removing season and trend

This separation helps you:

- Understand the underlying patterns in your data
- Identify anomalies that don't fit the patterns
- Make better forecasts by modeling each component separately
- Remove seasonality to focus on the trend

STL uses a technique called LOESS (locally estimated scatterplot smoothing) to
extract these components in a flexible way that works for many different types of data.

This class lets you configure exactly how the decomposition works to best match your data.

## How It Works

STL (Seasonal-Trend decomposition using Loess) is a robust method for decomposing time series data into 
three components: seasonal, trend, and remainder (residual). It uses iterative Loess smoothing (locally 
estimated scatterplot smoothing) to extract these components, making it flexible for handling a wide 
variety of seasonal patterns and trends. The algorithm is particularly valuable for time series with 
complex seasonality, non-linear trends, or outliers. This class provides extensive configuration options 
for controlling the STL decomposition process, including parameters for the seasonal and trend components, 
robustness iterations, window sizes, and additional adjustments for calendar effects. These options allow 
fine-tuning of the decomposition to match the specific characteristics of the time series being analyzed.

## Properties

| Property | Summary |
|:-----|:--------|
| `AdjustForDayOfWeek` | Gets or sets whether to adjust for day-of-week effects in the decomposition. |
| `AdjustForHolidays` | Gets or sets whether to adjust for holiday effects in the decomposition. |
| `AdjustForMonthOfYear` | Gets or sets whether to adjust for month-of-year effects in the decomposition. |
| `AlgorithmType` | Gets or sets the type of STL algorithm to use. |
| `Dates` | Gets or sets the array of dates corresponding to the time series observations. |
| `DayOfWeekFactors` | Gets or sets the factors for day-of-week effects. |
| `Holidays` | Gets or sets the dictionary of holiday dates and their effects. |
| `IQRMultiplier` | Gets or sets the multiplier for the interquartile range (IQR) when identifying outliers using the IQR method. |
| `InnerLoopPasses` | Gets or sets the number of passes through the inner loop of the STL algorithm. |
| `Interval` | Gets or sets the time interval between consecutive observations in the time series. |
| `LowPassBandwidth` | Gets or sets the bandwidth parameter for the low-pass filter LOESS smoothing. |
| `LowPassFilterWindowSize` | Gets or sets the window size for the low-pass filter. |
| `MonthOfYearFactors` | Gets or sets the factors for month-of-year effects. |
| `OutlierDetectionMethod` | Gets or sets the method used for detecting outliers in the time series. |
| `RobustIterations` | Gets or sets the number of robust iterations in the STL algorithm. |
| `RobustWeightThreshold` | Gets or sets the threshold for robust weights in outlier detection. |
| `SeasonalBandwidth` | Gets or sets the bandwidth parameter for the seasonal LOESS smoothing. |
| `SeasonalDegree` | Gets or sets the degree of the polynomial used in the seasonal LOESS smoothing. |
| `SeasonalJump` | Gets or sets the step size for the seasonal LOESS smoothing. |
| `SeasonalLoessWindow` | Gets or sets the window size for the seasonal component LOESS smoothing. |
| `SeasonalPeriod` | Gets or sets the number of time points in one seasonal cycle. |
| `StartDate` | Gets or sets the start date of the time series. |
| `TrendBandwidth` | Gets or sets the bandwidth parameter for the trend LOESS smoothing. |
| `TrendDegree` | Gets or sets the degree of the polynomial used in the trend LOESS smoothing. |
| `TrendJump` | Gets or sets the step size for the trend LOESS smoothing. |
| `TrendLoessWindow` | Gets or sets the window size for the trend LOESS smoothing. |
| `TrendWindowSize` | Gets or sets the window size for the trend component smoothing. |
| `ZScoreThreshold` | Gets or sets the threshold for identifying outliers using the Z-Score method. |

