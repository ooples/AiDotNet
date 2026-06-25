---
title: "GjrGarchModel<T>"
description: "GJR-GARCH(1,1,1) — Glosten, Jagannathan & Runkle (1993, \"On the Relation between the Expected Value and the Volatility of the Nominal Excess Return on Stocks\", Journal of Finance 48)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Volatility`

GJR-GARCH(1,1,1) — Glosten, Jagannathan & Runkle (1993, "On the Relation between the Expected Value
and the Volatility of the Nominal Excess Return on Stocks", Journal of Finance 48). Adds a LEVERAGE term
so negative shocks raise volatility more than positive ones:

Stationarity: α + γ/2 + β < 1; unconditional variance ω / (1 − α − γ/2 − β). Fit by maximum likelihood.

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

