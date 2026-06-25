---
title: "IPositionSizer<T>"
description: "A bet/position-sizing rule: given an edge, returns the fraction of capital to allocate."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Finance.Interfaces`

A bet/position-sizing rule: given an edge, returns the fraction of capital to allocate.

## For Beginners

A "position sizer" answers "how big should this bet be?" The default is the
Kelly criterion (the growth-optimal fraction); swap in your own rule if you prefer.

## How It Works

This is a customization point, not a trainable model. Trading agents and portfolio models depend on
this interface to decide *how much* to commit to a signal and default to the
`KellyCriterion` implementation, but callers can substitute
a fixed-fraction, volatility-target, or risk-budget sizer without changing the consuming model.

## Methods

| Method | Summary |
|:-----|:--------|
| `Continuous(,)` | Sizing fraction from the (Gaussian) mean and variance of returns. |
| `Discrete(,)` | Sizing fraction from a discrete win probability and win/loss payoff ratio. |
| `Fractional(,Double)` | Scales a base sizing fraction (e.g. |
| `FromReturns(IEnumerable<>,Double)` | Sizing fraction estimated from a realized return series. |

