---
title: "MarkowitzOptimizer<T>"
description: "Closed-form Markowitz mean-variance portfolio optimization: the global minimum-variance portfolio, the tangency (maximum-Sharpe) portfolio, and the efficient-frontier target-return portfolio."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Portfolio`

Closed-form Markowitz mean-variance portfolio optimization: the global minimum-variance portfolio,
the tangency (maximum-Sharpe) portfolio, and the efficient-frontier target-return portfolio.

## For Beginners

"Mean-variance" optimization picks how much to put in each asset by
trading off expected return against risk (variance), using how the assets move together (the
covariance matrix). The *minimum-variance* portfolio is the lowest-risk fully-invested mix and
ignores returns entirely. The *tangency* portfolio is the mix with the best risk-adjusted return
(highest Sharpe ratio) given a risk-free rate. *Target-return* finds the lowest-risk mix that
hits a return you specify. All three allow negative weights (short selling) and sum to 100%.

## How It Works

AiDotNet's portfolio optimizers are *neural* (they learn weights from market data through a
trained network). The classic analytic solutions — the ones you can write down with a covariance
inverse and no training — were missing. This fills that gap with the standard unconstrained
(long/short-allowed, fully-invested) closed forms. Each is just a linear solve against the covariance
matrix, reusing AiDotNet's `Inverse` rather than hand-rolling elimination.

## Properties

| Property | Summary |
|:-----|:--------|
| `Default` | Shared stateless default instance for injection as an `IMeanVarianceOptimizer`. |

## Methods

| Method | Summary |
|:-----|:--------|
| `IsUsable()` | True if a value is strictly positive and finite (net471-safe: no double.IsFinite). |
| `MinimumVariance(Matrix<>)` | Global minimum-variance portfolio: w = (Σ⁻¹·1) / (1ᵀ·Σ⁻¹·1). |
| `Normalize(Vector<>)` | Scales a weight vector so its entries sum to 1 (fully invested). |
| `RowSums(Matrix<>,Int32)` | Σ⁻¹·1 — the row sums of the inverse covariance (equivalently Σ⁻¹ times the all-ones vector). |
| `Tangency(Vector<>,Matrix<>,)` | Tangency (maximum-Sharpe) portfolio: w ∝ Σ⁻¹·(μ − rf·1), normalized to sum to 1. |
| `TargetReturn(Vector<>,Matrix<>,)` | Efficient-frontier portfolio for a given `targetReturn`, via the standard two-fund (Merton) Lagrangian: the minimum-variance fully-invested portfolio whose expected return equals the target. |
| `ValidateSquare(Matrix<>)` | Validates that the covariance is square and non-empty; returns its dimension N. |

