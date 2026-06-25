---
title: "LogNormalAFT<T>"
description: "Implements the Log-Normal Accelerated Failure Time (AFT) model for survival analysis."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SurvivalAnalysis`

Implements the Log-Normal Accelerated Failure Time (AFT) model for survival analysis.

## For Beginners

The Log-Normal AFT model assumes that the log of survival times
follows a normal distribution. This is appropriate when survival times have a bell-curve shape
after log-transformation, common in many biomedical and engineering applications.

## How It Works

**The Model:**
log(T) = β₀ + β₁X₁ + ... + βₚXₚ + σε
where ε ~ N(0,1) is standard normal.

**Interpretation:**

- exp(β) gives the multiplicative effect on median survival time
- A positive coefficient increases survival time (protective)
- A negative coefficient decreases survival time (harmful)

**When to use Log-Normal vs Weibull:**

- Log-Normal: Hazard first increases then decreases (non-monotonic)
- Weibull: Hazard is monotonic (increasing, decreasing, or constant)

**Reference:** Lawless (2003), Statistical Models and Methods for Lifetime Data

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LogNormalAFT(Int32,Double)` | Creates a new Log-Normal AFT model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Coefficients` | Gets the regression coefficients (β). |
| `Intercept` | Gets the intercept term (β₀ = μ). |
| `MaxIterations` | Gets the maximum iterations for optimization. |
| `Scale` | Gets the scale parameter (σ). |
| `Tolerance` | Gets the tolerance for convergence. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |
| `Deserialize(Byte[])` |  |
| `Erf(Double)` | Error function approximation (Abramowitz and Stegun). |
| `FitSurvivalCore(Matrix<>,Vector<>,Vector<Int32>)` | Fits the Log-Normal AFT model using maximum likelihood estimation. |
| `GetBaselineSurvival(Vector<>)` | Gets the baseline survival function. |
| `GetParameters` |  |
| `NormalCdf(Double)` | Standard normal CDF using error function approximation. |
| `NormalPdf(Double)` | Standard normal PDF. |
| `Predict(Matrix<>)` | Predicts median survival time. |
| `PredictHazardRatio(Matrix<>)` | Predicts acceleration factors (hazard ratios at baseline hazard). |
| `PredictSurvivalProbability(Matrix<>,Vector<>)` | Predicts survival probabilities at specified times. |
| `Serialize` |  |
| `SetParameters(Vector<>)` |  |
| `WithParameters(Vector<>)` |  |

