---
title: "TBATSModelOptions<T>"
description: "Configuration options for the TBATS (Trigonometric seasonality, Box-Cox transformation, ARMA errors,  Trend, and Seasonal components) time series forecasting model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the TBATS (Trigonometric seasonality, Box-Cox transformation, ARMA errors, 
Trend, and Seasonal components) time series forecasting model.

## For Beginners

TBATS is a powerful forecasting model for time series with complex seasonal patterns.

When forecasting time series data:

- Simple models struggle with multiple seasonal patterns (e.g., daily, weekly, and yearly cycles)
- Traditional approaches may require separate modeling for each seasonal pattern

TBATS solves this by:

- Handling multiple seasonal periods simultaneously
- Using Fourier series to efficiently represent seasonal patterns
- Transforming data to stabilize variance (Box-Cox transformation)
- Modeling short-term dependencies (ARMA components)
- Incorporating trend with optional damping

This approach offers several benefits:

- Effectively captures complex seasonal patterns
- Handles irregular seasonality (e.g., varying month lengths)
- Produces accurate forecasts for data with multiple cycles
- Automatically selects appropriate components

This class lets you configure how the TBATS model analyzes and forecasts your time series data.

## How It Works

TBATS is an advanced time series forecasting model designed to handle complex seasonal patterns, including 
multiple seasonal periods. It combines several powerful techniques: Box-Cox transformation for stabilizing 
variance, Fourier terms for handling multiple seasonal patterns, ARMA models for capturing short-term 
dependencies, and trend components with optional damping. TBATS is particularly effective for time series 
with multiple seasonal patterns of different lengths (e.g., daily, weekly, and yearly patterns) and can 
automatically select the appropriate components based on the data. This class provides configuration options 
for controlling the various components and optimization parameters of the TBATS model.

## Properties

| Property | Summary |
|:-----|:--------|
| `ARMAOrder` | Gets or sets the order of the ARMA (AutoRegressive Moving Average) component. |
| `BoxCoxLambda` | Gets or sets the Box-Cox transformation parameter. |
| `DecompositionType` | Gets or sets the type of matrix decomposition used in the optimization algorithm. |
| `MaxIterations` | Gets or sets the maximum number of iterations for the optimization algorithm. |
| `SeasonalPeriods` | Gets or sets the seasonal periods to model. |
| `Tolerance` | Gets or sets the convergence tolerance for the optimization algorithm. |
| `TrendDampingFactor` | Gets or sets the damping factor for the trend component. |

