---
title: "WeibullAFT<T>"
description: "Implements the Weibull Accelerated Failure Time (AFT) model for survival analysis."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SurvivalAnalysis`

Implements the Weibull Accelerated Failure Time (AFT) model for survival analysis.

## For Beginners

The Weibull AFT model assumes survival times follow a Weibull distribution.
Unlike Cox models that model hazard ratios, AFT models directly model how covariates "accelerate"
or "decelerate" time to event. Coefficients can be interpreted as the effect on survival time.

## How It Works

**The Model:**
log(T) = β₀ + β₁X₁ + ... + βₚXₚ + σε
where T is survival time, X are covariates, β are coefficients, σ is scale, and ε ~ extreme value distribution.

**Interpretation:**

- A positive coefficient means longer survival (protective effect)
- A negative coefficient means shorter survival (risk factor)
- exp(β) gives the multiplicative effect on survival time

**Weibull distribution:**

- Shape parameter κ controls hazard shape (κ < 1: decreasing, κ = 1: constant, κ > 1: increasing)
- Scale parameter λ controls time scale

**Reference:** Lawless (2003), Statistical Models and Methods for Lifetime Data

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `WeibullAFT(Int32,Double)` | Creates a new Weibull AFT model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Coefficients` | Gets the regression coefficients (β). |
| `Intercept` | Gets the intercept term (β₀). |
| `MaxIterations` | Gets the maximum iterations for optimization. |
| `Scale` | Gets the scale parameter (σ). |
| `Shape` | Gets the shape parameter (κ = 1/σ). |
| `Tolerance` | Gets the tolerance for convergence. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |
| `Deserialize(Byte[])` |  |
| `FitSurvivalCore(Matrix<>,Vector<>,Vector<Int32>)` | Fits the Weibull AFT model by maximizing the log-likelihood. |
| `GetBaselineSurvival(Vector<>)` | Gets the baseline survival function. |
| `GetParameters` |  |
| `Predict(Matrix<>)` | Predicts median survival time. |
| `PredictHazardRatio(Matrix<>)` | Predicts hazard ratios (acceleration factors) relative to baseline. |
| `PredictSurvivalProbability(Matrix<>,Vector<>)` | Predicts survival probabilities at specified times. |
| `Serialize` |  |
| `SetParameters(Vector<>)` |  |
| `WithParameters(Vector<>)` |  |

