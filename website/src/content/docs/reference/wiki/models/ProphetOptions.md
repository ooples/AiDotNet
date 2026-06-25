---
title: "ProphetOptions<T, TInput, TOutput>"
description: "Configuration options for Prophet, a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Prophet, a procedure for forecasting time series data based on an additive model
where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects.

## For Beginners

Prophet is a powerful tool for predicting future values in time-based data.

Imagine you're trying to forecast your company's sales for the next few months:

- Sales might follow patterns that repeat yearly (holiday shopping seasons)
- They might follow weekly patterns (higher on weekends, lower on Mondays)
- There might be special days that affect sales (Black Friday, Cyber Monday)
- The overall trend might be growing, shrinking, or changing direction occasionally

What Prophet does:

- It breaks down your data into separate pieces (components):
- Base trend: The overall direction (growing or declining)
- Seasonality: Repeating patterns (yearly, weekly, daily)
- Holiday effects: Impacts of special days
- Changepoints: Where the trend changes direction

Think of it like weather forecasting:

- It looks at historical patterns
- It accounts for seasons and special events
- It combines these patterns to predict what will happen next

The benefit of Prophet is that it automatically handles many complex aspects of time series forecasting
that would otherwise require significant expertise. This class lets you configure how Prophet analyzes 
and forecasts your time series data.

## How It Works

Prophet is a forecasting procedure developed by Facebook (now Meta) that decomposes time series into several components:
trend, seasonality, holiday effects, and error. It is designed to handle time series with strong seasonal patterns, 
missing values, outliers, and shifts in the trend. Prophet works particularly well with time series that have
strong seasonal effects and several seasons of historical data. The model automatically detects changes in trends 
by selecting changepoints from the data, while also allowing manual specification of known changepoints. 
Prophet is robust to missing data and shifts in the trend, and typically handles outliers well. 
It is particularly useful for business and economic time series that are affected by holidays and seasonal patterns.

## Properties

| Property | Summary |
|:-----|:--------|
| `AnomalyThresholdSigma` | Gets or sets the number of standard deviations from the mean residual to use as the anomaly threshold. |
| `ApplyTransformation` | Gets or sets whether to apply transformations to the predictions. |
| `ChangePointPriorScale` | Gets or sets the flexibility of the trend changepoints. |
| `ComputePredictionIntervals` | Gets or sets a value indicating whether to compute prediction intervals. |
| `DailySeasonality` | Gets or sets a value indicating whether daily seasonality should be included in the model. |
| `EnableAnomalyDetection` | Gets or sets a value indicating whether to enable anomaly detection during and after training. |
| `ForecastHorizon` | Gets or sets the number of periods to forecast into the future. |
| `FourierOrder` | Gets or sets the number of Fourier terms used for modeling seasonality. |
| `HolidayPriorScale` | Gets or sets the strength of the holiday effects. |
| `InitialChangepointValue` | Gets or sets the initial value for changepoint effects. |
| `InitialTrendValue` | Gets or sets the initial value for the trend component. |
| `OptimizeParameters` | Gets or sets a value indicating whether the model should attempt to optimize its parameters during training. |
| `Optimizer` | Gets or sets the optimizer to use for parameter fitting. |
| `PredictionIntervalWidth` | Gets or sets the confidence level for prediction intervals. |
| `RegressorCount` | Gets or sets the number of additional regressor variables to include in the model. |
| `SeasonalityPriorScale` | Gets or sets the strength of the seasonality components. |
| `TransformPrediction` | Gets or sets the function used to transform predictions. |
| `WeeklySeasonality` | Gets or sets a value indicating whether weekly seasonality should be included in the model. |
| `YearlySeasonality` | Gets or sets a value indicating whether yearly seasonality should be included in the model. |

