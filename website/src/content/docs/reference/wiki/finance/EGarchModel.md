---
title: "EGarchModel<T>"
description: "EGARCH(1,1) — Nelson (1991, \"Conditional Heteroskedasticity in Asset Returns: A New Approach\", Econometrica 59)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Volatility`

EGARCH(1,1) — Nelson (1991, "Conditional Heteroskedasticity in Asset Returns: A New Approach",
Econometrica 59). Models LOG conditional variance, so positivity holds automatically and the leverage
effect enters linearly in the standardized shock z = ε/σ:

with E|z| = √(2/π) for Gaussian innovations, and |β| < 1 for stationarity. γ < 0 captures the
leverage effect (negative returns raise vol more). Fit by maximum likelihood.

## Properties

| Property | Summary |
|:-----|:--------|
| `ModelName` |  |
| `ParameterCount` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateInstance` |  |
| `InitialGuess(Double)` |  |
| `MeanReversionSpeed(Double[])` |  |
| `NextVariance(Double,Double,Double[])` |  |
| `ToNatural(Double[])` |  |
| `ToUnconstrained(Double[])` |  |
| `UnconditionalVariance(Double[])` |  |

