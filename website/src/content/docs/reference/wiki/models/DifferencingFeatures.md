---
title: "DifferencingFeatures"
description: "Flags for selecting which differencing and stationarity features to compute."
section: "API Reference"
---

`Enums` · `AiDotNet.Models.Options`

Flags for selecting which differencing and stationarity features to compute.

## For Beginners

Differencing transforms help make time series stationary,
which is required by many forecasting models. A stationary series has:

- Constant mean over time (no trend)
- Constant variance over time (no changing volatility)
- No seasonal patterns

## Fields

| Field | Summary |
|:-----|:--------|
| `All` | All available differencing features. |
| `BasicDifferencing` | All basic differencing methods. |
| `Decomposition` | All decomposition methods. |
| `Detrending` | All detrending methods. |
| `FirstDifference` | First-order differencing: y[t] - y[t-1]. |
| `HodrickPrescottFilter` | Hodrick-Prescott filter: extracts trend and cycle components. |
| `LinearDetrend` | Linear detrending: removes best-fit straight line. |
| `LogDifference` | Log difference: log(y[t]) - log(y[t-1]). |
| `None` | No differencing features. |
| `PercentChange` | Percent change: (y[t] - y[t-1]) / y[t-1]. |
| `PolynomialDetrend` | Polynomial detrending: removes best-fit polynomial. |
| `Returns` | All return-based transforms. |
| `SeasonalDifference` | Seasonal differencing: y[t] - y[t-period]. |
| `SecondDifference` | Second-order differencing: diff(diff(y)). |
| `StlDecomposition` | STL decomposition: Seasonal-Trend decomposition using LOESS. |

