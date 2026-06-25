---
title: "IRiskRatioCalculator<T>"
description: "Computes risk-adjusted performance ratios (Sharpe, Sortino, Calmar) from a periodic return series."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Finance.Interfaces`

Computes risk-adjusted performance ratios (Sharpe, Sortino, Calmar) from a periodic return series.

## For Beginners

These ratios all divide "how much you earned" by "how much risk you took";
higher is better. The default implementation uses the standard textbook formulas.

## How It Works

This is a customization point, not a trainable model. Risk models, backtests, and trading agents
depend on this interface to score a strategy's risk-adjusted return and default to the
`RiskRatios` implementation, but callers can substitute their
own conventions (e.g. a different annualization, a downside threshold other than zero).

## Methods

| Method | Summary |
|:-----|:--------|
| `Calmar(IReadOnlyList<>,Int32)` | Calmar ratio (annualized return / maximum drawdown). |
| `Sharpe(IReadOnlyList<>,Double,Int32)` | Annualized Sharpe ratio (mean excess return / total volatility). |
| `Sortino(IReadOnlyList<>,Double,Int32)` | Annualized Sortino ratio (mean excess return / downside deviation). |

