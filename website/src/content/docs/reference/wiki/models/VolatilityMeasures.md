---
title: "VolatilityMeasures"
description: "Flags for selecting which volatility measures to calculate."
section: "API Reference"
---

`Enums` · `AiDotNet.Models.Options`

Flags for selecting which volatility measures to calculate.

## Fields

| Field | Summary |
|:-----|:--------|
| `Advanced` | Advanced volatility models (EWMA, GARCH). |
| `All` | All volatility measures. |
| `Basic` | Basic volatility measures (returns, realized vol, momentum). |
| `EwmaVolatility` | EWMA (Exponentially Weighted Moving Average) volatility. |
| `GarchVolatility` | GARCH(1,1) volatility estimator. |
| `GarmanKlassVolatility` | Garman-Klass volatility (OHLC based). |
| `LogReturns` | Log returns (ln(price / previous price)). |
| `Momentum` | Price momentum (current price / past price - 1). |
| `None` | No volatility measures. |
| `OhlcBased` | OHLC-based volatility estimators. |
| `ParkinsonVolatility` | Parkinson volatility (high-low range based). |
| `RealizedVolatility` | Realized volatility (standard deviation of returns). |
| `RogersSatchellVolatility` | Rogers-Satchell volatility estimator. |
| `SimpleReturns` | Simple returns (price change / previous price). |
| `YangZhangVolatility` | Yang-Zhang volatility estimator using OHLC data. |

