---
title: "KellyCriterion<T>"
description: "Kelly-criterion position sizing — the bet fraction that maximizes long-run log-growth of capital."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Portfolio`

Kelly-criterion position sizing — the bet fraction that maximizes long-run log-growth of capital.

## For Beginners

Kelly tells you what fraction of your capital to put on a trade to grow
wealth fastest over many trades. Bet more than Kelly and you risk ruin; bet a fraction of it
("half-Kelly") for much lower volatility at a small growth cost. A negative Kelly means "no edge —
don't take the trade."

## How It Works

AiDotNet's portfolio optimizers cover weight allocation (mean-variance, HRP, Black-Litterman) but
not Kelly bet sizing. This fills that gap with the two standard forms — discrete (win probability +
payoff odds) and continuous (mean / variance of returns) — plus fractional Kelly, which is what
practitioners actually use because full Kelly is famously over-aggressive.

## Properties

| Property | Summary |
|:-----|:--------|
| `Default` | Shared stateless default instance for injection as an `IPositionSizer`. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Continuous(,)` | Continuous Kelly fraction for (approximately) Gaussian returns: f* = μ / σ², where `expectedReturn` is μ and `variance` is σ². |
| `Discrete(,)` | Discrete Kelly fraction: f* = p − (1 − p) / b, where `winProbability` is p and `winLossRatio` is b (net payoff on a win per unit risked on a loss). |
| `Fractional(,Double)` | Fractional Kelly: `kellyFraction` scaled by `fraction` (e.g. |
| `FromReturns(IEnumerable<>,Double)` | Continuous Kelly estimated from a return series: computes the sample mean and (population) variance of `returns` and applies `Continuous(`. |

