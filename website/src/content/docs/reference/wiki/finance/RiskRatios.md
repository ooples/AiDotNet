---
title: "RiskRatios<T>"
description: "Risk-adjusted performance ratios from a periodic return series: Sharpe, Sortino, and Calmar."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Risk`

Risk-adjusted performance ratios from a periodic return series: Sharpe, Sortino, and Calmar.

## For Beginners

Sharpe divides average return by total volatility; Sortino only counts
*downside* volatility (it doesn't punish upside swings); Calmar divides annualized return by the
worst peak-to-trough drawdown. Higher is better for all three.

## How It Works

AiDotNet exposes Sharpe on portfolio optimizers and trading agents, but Sortino (downside-only) and
Calmar (return-over-max-drawdown) were missing as standalone, series-based utilities. These let any
forecaster/strategy backtest be scored on risk-adjusted return without a portfolio or agent object.

## Properties

| Property | Summary |
|:-----|:--------|
| `Default` | Shared stateless default instance for injection as an `IRiskRatioCalculator`. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calmar(IReadOnlyList<>,Int32)` | Calmar ratio: annualized return / maximum drawdown, computed from a periodic return series by compounding an equity curve. |
| `Sharpe(IReadOnlyList<>,Double,Int32)` | Annualized Sharpe ratio: (mean(excess) / std(excess)) · √periodsPerYear. |
| `Sortino(IReadOnlyList<>,Double,Int32)` | Annualized Sortino ratio: (mean(excess) / downsideDeviation) · √periodsPerYear. |
| `Validate(IReadOnlyList<>,Int32)` | Validates the shared external inputs of every ratio. |

