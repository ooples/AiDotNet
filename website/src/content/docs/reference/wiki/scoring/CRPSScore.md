---
title: "CRPSScore<T>"
description: "Continuous Ranked Probability Score (CRPS) scoring rule."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Scoring`

Continuous Ranked Probability Score (CRPS) scoring rule.

## For Beginners

CRPS rewards predictions that are both accurate (close to
the true value) and confident (narrow distributions). Unlike log score, CRPS:

- Has the same units as the predicted variable (like MAE)
- Is robust to outliers
- Considers the full shape of the distribution, not just probability at one point

For point forecasts, CRPS reduces to mean absolute error.

## How It Works

CRPS measures the integral of the squared difference between the predicted
cumulative distribution function (CDF) and the empirical CDF of the observation.
It generalizes the mean absolute error to probabilistic predictions.

CRPS = ∫ (F(x) - 1(x ≥ y))² dx
where F is the predicted CDF and y is the observation.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CRPSScore(Int32)` | Initializes a new CRPS scoring rule. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsMinimized` |  |
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Score(IParametricDistribution<>,)` |  |
| `ScoreGradient(IParametricDistribution<>,)` |  |
| `ScoreGradientNormal(NormalDistribution<>,)` | Analytical CRPS gradient for Normal distribution. |
| `ScoreGradientNumerical(IParametricDistribution<>,)` | Numerical CRPS gradient computation. |
| `ScoreLaplace(LaplaceDistribution<>,)` | Closed-form CRPS for Laplace distribution. |
| `ScoreNormal(NormalDistribution<>,)` | Closed-form CRPS for Normal distribution. |
| `ScoreNumerical(IParametricDistribution<>,)` | Numerical CRPS computation using integration. |

