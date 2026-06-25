---
title: "IVolatilityModel<T>"
description: "Interface for volatility models that forecast price variability."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Finance.Interfaces`

Interface for volatility models that forecast price variability.

## For Beginners

Volatility measures how "bouncy" prices are.

**Why Volatility Matters:**

- Options pricing: Higher volatility = more expensive options
- Risk management: Volatile assets need more capital buffer
- Portfolio construction: Helps balance risk across assets

**Types of Volatility:**

- **Historical:** What happened in the past
- **Implied:** What the market expects (from option prices)
- **Realized:** What actually occurred over a period
- **Forecast:** Our prediction for the future

## How It Works

Volatility models predict how much prices will fluctuate, essential for
option pricing, risk management, and portfolio construction.

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateRealizedVolatility(Tensor<>)` | Calculates realized volatility from high-frequency data. |
| `ComputeCorrelationMatrix(Tensor<>)` | Computes the correlation matrix from returns data. |
| `ComputeCovarianceMatrix(Tensor<>)` | Computes the covariance matrix from returns data. |
| `EstimateCurrentVolatility(Tensor<>)` | Estimates the current volatility state. |
| `ForecastVolatility(Tensor<>,Int32)` | Forecasts future volatility. |
| `GetVolatilityMetrics` | Gets volatility-specific metrics. |

