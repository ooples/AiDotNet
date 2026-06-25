---
title: "IMeanVarianceOptimizer<T>"
description: "A closed-form mean-variance portfolio optimizer: the analytic (training-free) weight solutions."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Finance.Interfaces`

A closed-form mean-variance portfolio optimizer: the analytic (training-free) weight solutions.

## For Beginners

"Mean-variance" optimization divides money across assets by trading off
expected return against risk. These are the textbook closed-form answers; the default implementation
is the standard Markowitz solver.

## How It Works

This is a customization point, not a trainable model — unlike `IPortfolioOptimizer`
(the neural, data-fit family), this exposes the classic closed-form Markowitz solutions a portfolio
model can use directly as a baseline or warm start. Consumers default to
`MarkowitzOptimizer` but can substitute their own analytic
solver (shrinkage covariance, constrained QP, …).

## Methods

| Method | Summary |
|:-----|:--------|
| `MinimumVariance(Matrix<>)` | Global minimum-variance fully-invested weights (expected returns ignored). |
| `Tangency(Vector<>,Matrix<>,)` | Tangency (maximum-Sharpe) fully-invested weights for the given risk-free rate. |
| `TargetReturn(Vector<>,Matrix<>,)` | Minimum-variance fully-invested weights achieving a target expected return. |

