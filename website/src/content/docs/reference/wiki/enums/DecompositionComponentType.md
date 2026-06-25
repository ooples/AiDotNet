---
title: "DecompositionComponentType"
description: "Represents the different components that can be extracted when decomposing a time series."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Represents the different components that can be extracted when decomposing a time series.

## For Beginners

Time series decomposition is like breaking down a complex song into its individual instruments.

When analyzing data that changes over time (like stock prices, temperature readings, or website traffic),
it's often helpful to separate the data into simpler components to better understand what's happening.

For example, retail sales data might contain:

- A general upward trend due to business growth
- Seasonal patterns (higher sales during holidays)
- Random fluctuations due to unpredictable factors

Decomposing the data helps you see each of these patterns separately, making it easier to:

- Understand what's driving changes in your data
- Make better forecasts
- Identify unusual events or anomalies

## Fields

| Field | Summary |
|:-----|:--------|
| `Cycle` | Repeating patterns with a variable or changing period, unlike the fixed periods of seasonal components. |
| `IMF` | Intrinsic Mode Functions - components extracted using Empirical Mode Decomposition (EMD) methods. |
| `Irregular` | Random, unpredictable fluctuations in the data (similar to Residual but used in specific decomposition methods). |
| `Residual` | The irregular variation or "noise" remaining after other components have been extracted. |
| `Seasonal` | Repeating patterns or cycles with a fixed, known period (e.g., daily, weekly, monthly, yearly). |
| `Trend` | The long-term progression or general direction of the time series. |
| `TrendCycle` | A combined component that includes both the long-term trend and cyclical patterns. |

