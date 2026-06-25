---
title: "SEATSAlgorithmType"
description: "Represents different algorithm types for SEATS (Seasonal Extraction in ARIMA Time Series) decomposition."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums.AlgorithmTypes`

Represents different algorithm types for SEATS (Seasonal Extraction in ARIMA Time Series) decomposition.

## For Beginners

SEATS (Seasonal Extraction in ARIMA Time Series) is a method used to break down time series data 
into different components, making it easier to understand patterns and make predictions.

Think of time series data as a recording of values over time, like daily temperature readings, monthly sales figures, 
or quarterly economic indicators. This data often contains several patterns mixed together:

1. Trend: The long-term direction (going up, down, or staying flat)
2. Seasonal patterns: Regular fluctuations that repeat at fixed intervals (like higher sales during holidays)
3. Cyclical patterns: Longer-term ups and downs (like business cycles)
4. Irregular components: Random fluctuations that don't follow any pattern

SEATS helps separate these components by:

1. Modeling the time series using ARIMA (AutoRegressive Integrated Moving Average) methods
2. Identifying and extracting the seasonal patterns
3. Separating the trend from the irregular components

Why is SEATS important in AI and machine learning?

1. Improved Forecasting: By understanding each component separately, predictions become more accurate

2. Pattern Recognition: Helps AI systems identify meaningful patterns versus random noise

3. Anomaly Detection: Makes it easier to spot unusual events that don't fit established patterns

4. Feature Engineering: Creates useful features for machine learning models from time series data

5. Seasonal Adjustment: Allows for fair comparisons between different time periods by removing seasonal effects

This enum specifies which specific algorithm variant to use for SEATS decomposition, as different methods have 
different characteristics and may be more suitable for certain types of time series data.

## Fields

| Field | Summary |
|:-----|:--------|
| `Burman` | Uses Burman's variant of the SEATS algorithm, which focuses on robust seasonal adjustment. |
| `Canonical` | Uses the canonical SEATS decomposition approach, which enforces specific constraints on the components. |
| `Standard` | Uses the standard SEATS algorithm for time series decomposition. |

