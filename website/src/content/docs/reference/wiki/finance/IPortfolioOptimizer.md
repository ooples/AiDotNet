---
title: "IPortfolioOptimizer<T>"
description: "Interface for portfolio optimization models that determine optimal asset allocations."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Finance.Interfaces`

Interface for portfolio optimization models that determine optimal asset allocations.

## For Beginners

Portfolio optimization answers: "How should I divide my money?"

**Key Concepts:**

- **Expected Return:** How much you expect to earn
- **Risk (Volatility):** How much prices bounce around
- **Diversification:** Don't put all eggs in one basket

**Common Objectives:**

- Maximize returns for given risk (Sharpe ratio)
- Minimize risk for given return (minimum variance)
- Risk parity (equal risk contribution)
- Maximum diversification

## How It Works

Portfolio optimizers determine how to allocate capital across assets to achieve
investment objectives while managing risk.

## Properties

| Property | Summary |
|:-----|:--------|
| `NumAssets` | Gets the number of assets in the portfolio universe. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateExpectedReturn(Tensor<>,Tensor<>)` | Calculates expected portfolio return. |
| `CalculateSharpeRatio(Tensor<>,Tensor<>,Tensor<>,)` | Calculates the Sharpe ratio of the portfolio. |
| `CalculateVolatility(Tensor<>,Tensor<>)` | Calculates portfolio volatility (standard deviation of returns). |
| `ComputeRiskContribution(Tensor<>,Tensor<>)` | Computes portfolio risk contribution for each asset. |
| `GetPortfolioMetrics` | Gets portfolio-specific performance metrics. |
| `OptimizeWeights(Tensor<>,Tensor<>)` | Optimizes portfolio weights given expected returns and covariance. |

