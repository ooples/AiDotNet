---
title: "Garch11Model<T>"
description: "GARCH(1,1) — Bollerslev (1986, \"Generalized Autoregressive Conditional Heteroskedasticity\", Journal of Econometrics 31)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Volatility`

GARCH(1,1) — Bollerslev (1986, "Generalized Autoregressive Conditional Heteroskedasticity",
Journal of Econometrics 31). Conditional variance:

with ω > 0, α ≥ 0, β ≥ 0 and α + β < 1 (covariance-stationary). Fit by maximum likelihood.
The unconditional variance is ω / (1 − α − β). The workhorse volatility model.

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

