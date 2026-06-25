---
title: "GrangerCausalityAlgorithm<T>"
description: "Granger Causality — time series causal discovery via predictive improvement."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.TimeSeries`

Granger Causality — time series causal discovery via predictive improvement.

## For Beginners

Imagine predicting tomorrow's temperature. If knowing yesterday's
humidity helps predict temperature better than just knowing past temperatures alone,
then humidity "Granger-causes" temperature. This doesn't prove true causation but
indicates a useful predictive relationship.

## How It Works

Granger causality tests whether the past values of one variable X improve the prediction
of another variable Y beyond what Y's own past values provide. If so, X "Granger-causes" Y.

**Test procedure for each pair (i → j):**

- Fit a restricted model: Y_t = f(Y_{t-1}, ..., Y_{t-L}) — autoregressive on Y only
- Fit an unrestricted model: Y_t = f(Y_{t-1}, ..., Y_{t-L}, X_{t-1}, ..., X_{t-L})
- Compare using F-test: F = ((RSS_r - RSS_u) / L) / (RSS_u / (n - 2L - 1))
- If F is significant (p < alpha), X Granger-causes Y

Reference: Granger (1969), "Investigating Causal Relations by Econometric Models
and Cross-spectral Methods", Econometrica.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GrangerCausalityAlgorithm(CausalDiscoveryOptions)` | Initializes Granger Causality with optional configuration. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsNonlinear` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeLaggedCrossCorrelation(Matrix<>,Int32,Int32,Int32)` | Computes lagged cross-correlation between cause (at lag) and target (at present). |
| `ContinuedFractionBeta(Double,Double,Double)` | Evaluates the continued fraction for the incomplete beta function. |
| `DiscoverStructureCore(Matrix<>)` |  |
| `FDistributionSurvivalFunction(Double,Int32,Int32)` | Computes 1 - F_CDF(x; d1, d2) using the regularized incomplete beta function. |
| `LogGamma(Double)` | Computes log(Gamma(x)) using the Lanczos approximation. |
| `RegularizedIncompleteBeta(Double,Double,Double)` | Computes the regularized incomplete beta function I_x(a, b) using a continued fraction expansion. |

