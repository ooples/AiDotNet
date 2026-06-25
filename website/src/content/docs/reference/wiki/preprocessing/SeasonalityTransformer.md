---
title: "SeasonalityTransformer<T>"
description: "Generates seasonality and calendar features for time series data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.TimeSeries`

Generates seasonality and calendar features for time series data.

## For Beginners

Many real-world patterns repeat over time:

- **Daily patterns**: Energy usage peaks in morning/evening
- **Weekly patterns**: Retail sales higher on weekends
- **Monthly patterns**: Bill payments cluster at month start/end
- **Yearly patterns**: Tourism peaks in summer, retail peaks in December

This transformer creates numerical features that help ML models learn these patterns.
For example, instead of just knowing "it's Monday", the model gets:

- Day of week = 1 (Monday)
- Is weekend = 0 (no)
- Sin/Cos waves that smoothly encode the position in the week

## How It Works

This transformer creates features that capture time-based patterns including:

- Fourier features for smooth cyclical patterns
- Calendar features (hour, day, week, month, quarter, year)
- Holiday and event features
- Trading day features for financial data

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SeasonalityTransformer(TimeSeriesFeatureOptions)` | Creates a new seasonality transformer with the specified options. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SeasonalityFeatureCount` | Gets the number of output features per time step. |
| `SupportsInverseTransform` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeCalendarEventFeatures(DateTime,Tensor<>,Int32,Int32)` | Computes calendar event features. |
| `ComputeFourierFeatures(Int32,Int32,Tensor<>,Int32,Int32)` | Computes Fourier features (sin/cos at seasonal frequencies). |
| `ComputeIncrementalFeatures(IncrementalState<>,[])` | Computes seasonality features incrementally based on time index. |
| `ComputeIndexBasedFeatures(Int32,Int32,Tensor<>,Int32,Int32)` | Computes index-based features when no date is available. |
| `ComputeTimeFeatures(DateTime,Tensor<>,Int32,Int32)` | Computes time-based features from the date. |
| `ComputeTradingFeatures(DateTime,Int32,Tensor<>,Int32,Int32)` | Computes trading-specific features. |
| `ExportParameters` | Exports transformer-specific parameters for serialization. |
| `FitCore(Tensor<>)` |  |
| `GenerateFeatureNames` |  |
| `GetDateForTimeStep(Int32)` | Gets the date for a given time step. |
| `GetOperationNames` |  |
| `GetTradingDayOfMonth(DateTime)` | Calculates the trading day of the month (skips weekends). |
| `GetTradingDayOfWeek(DateTime)` | Gets the trading day of the week (1-5, Monday-Friday). |
| `ImportParameters(Dictionary<String,Object>)` | Imports transformer-specific parameters for validation. |
| `TransformCore(Tensor<>)` |  |
| `TransformParallel(Tensor<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_enabledFeatures` | The enabled seasonality features. |
| `_featureNames` | Cached feature names. |
| `_fourierTerms` | Number of Fourier terms per period. |
| `_holidayDates` | Holiday dates for holiday features. |
| `_holidayWindowDays` | Window days around holidays. |
| `_interval` | Time interval between data points. |
| `_isTradingDayData` | Whether data represents trading days only. |
| `_seasonalPeriods` | Seasonal periods for Fourier features. |
| `_startDate` | Start date of the time series. |

