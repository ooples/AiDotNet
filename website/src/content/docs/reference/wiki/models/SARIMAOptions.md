---
title: "SARIMAOptions<T>"
description: "Configuration options for Seasonal Autoregressive Integrated Moving Average (SARIMA) models, which extend ARIMA models to incorporate seasonal components in time series data."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Seasonal Autoregressive Integrated Moving Average (SARIMA) models,
which extend ARIMA models to incorporate seasonal components in time series data.

## For Beginners

SARIMA helps predict future values in time series data that has seasonal patterns.

Time series data shows how values change over time, like:

- Monthly sales figures
- Daily temperature readings
- Quarterly company earnings

Many time series have seasonal patterns:

- Retail sales spike during holidays
- Ice cream consumption increases in summer
- Energy usage follows daily and yearly cycles

SARIMA is designed to capture both:

- The overall trend in your data
- The repeating seasonal patterns

It does this by combining:

- Regular ARIMA components that handle short-term patterns and trends
- Seasonal components that handle repeating patterns at fixed intervals

This class lets you configure exactly how the SARIMA model analyzes your time series data,
including how many past values to consider and how to handle seasonality.

## How It Works

SARIMA (Seasonal Autoregressive Integrated Moving Average) is a sophisticated time series forecasting model 
that extends the ARIMA (Autoregressive Integrated Moving Average) framework to handle seasonal patterns in data. 
It is particularly useful for time series that exhibit both trend and seasonality, such as monthly sales data, 
quarterly economic indicators, or daily web traffic with weekly patterns. The model is denoted as 
SARIMA(p,d,q)(P,D,Q)m, where p, d, q are the non-seasonal parameters (autoregressive order, differencing order, 
and moving average order), P, D, Q are the corresponding seasonal parameters, and m is the seasonal period. 
This class provides configuration options for all these parameters, allowing fine-tuning of the SARIMA model 
to best capture the specific characteristics of the time series being analyzed.

## Properties

| Property | Summary |
|:-----|:--------|
| `D` | Gets or sets the differencing order (d) of the non-seasonal component. |
| `MaxIterations` | Gets or sets the maximum number of iterations for the parameter estimation algorithm. |
| `P` | Gets or sets the autoregressive order (p) of the non-seasonal component. |
| `Q` | Gets or sets the moving average order (q) of the non-seasonal component. |
| `SeasonalD` | Gets or sets the seasonal differencing order (D) of the model. |
| `SeasonalP` | Gets or sets the seasonal autoregressive order (P) of the model. |
| `SeasonalPeriod` | Gets or sets the number of time points in one seasonal cycle. |
| `SeasonalQ` | Gets or sets the seasonal moving average order (Q) of the model. |
| `Tolerance` | Gets or sets the convergence tolerance for the parameter estimation algorithm. |

