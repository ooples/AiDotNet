---
title: "CoxProportionalHazards<T>"
description: "Implements the Cox Proportional Hazards model for survival analysis with covariates."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SurvivalAnalysis`

Implements the Cox Proportional Hazards model for survival analysis with covariates.

## For Beginners

Cox Proportional Hazards is the most widely used survival model
because it can tell you HOW features affect survival risk without making assumptions
about the shape of survival over time.

The key equation is:
h(t|X) = h₀(t) × exp(β₁X₁ + β₂X₂ + ...)

Where:

- h(t|X) = hazard (instantaneous risk) at time t for a subject with features X
- h₀(t) = baseline hazard (unknown, estimated non-parametrically)
- β = coefficients that tell you how each feature affects risk
- X = feature values

Interpreting coefficients:

- β > 0: Higher feature value → Higher risk (shorter survival)
- β < 0: Higher feature value → Lower risk (longer survival)
- exp(β) = Hazard Ratio: How much risk changes per unit increase in feature

Example: If β_age = 0.05, then exp(0.05) ≈ 1.05, meaning each year of age
increases the hazard (risk) by about 5%.

The "proportional hazards" assumption means that hazard ratios are constant over time.
If this assumption is violated, consider stratified Cox or time-varying coefficients.

References:

- Cox (1972). "Regression Models and Life-Tables"

## How It Works

The Cox model is a semi-parametric survival model that estimates the effect of covariates
on the hazard (risk) without assuming a specific form for the baseline hazard function.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CoxProportionalHazards(Double,Int32,Double,Double,Nullable<Int32>)` | Gets the model type. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeBaselineSurvival(Matrix<>,Vector<>,Vector<Int32>)` | Computes the Breslow baseline survival function. |
| `ComputeGradient(Matrix<>,Vector<>,Vector<Int32>,Int32[])` | Computes the gradient of the partial log-likelihood. |
| `CreateNewInstance` | Creates a new instance of the same type. |
| `FitSurvivalCore(Matrix<>,Vector<>,Vector<Int32>)` | Fits the Cox model using partial likelihood maximization. |
| `GetBaselineSurvival(Vector<>)` | Gets the baseline survival function at specified time points. |
| `GetCoefficients` | Gets the estimated coefficients (log hazard ratios). |
| `GetFeatureHazardRatios` | Gets the hazard ratios for each feature. |
| `GetFeatureImportance` | Gets the feature importance scores based on coefficient magnitudes. |
| `GetParameters` | Gets all model parameters as a single vector. |
| `Predict(Matrix<>)` | Standard prediction - returns hazard ratios. |
| `PredictHazardRatio(Matrix<>)` | Predicts hazard ratios for each subject. |
| `PredictSurvivalProbability(Matrix<>,Vector<>)` | Predicts survival probabilities at specified time points. |
| `SetParameters(Vector<>)` | Sets the parameters for this model. |
| `WithParameters(Vector<>)` | Creates a new instance of the model with specified parameters. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_coefficients` | The estimated coefficients (log hazard ratios). |
| `_l2Penalty` | L2 regularization strength (ridge penalty). |
| `_learningRate` | The learning rate for gradient descent. |
| `_maxIterations` | Maximum number of iterations for optimization. |
| `_random` | Random number generator for initialization. |
| `_tolerance` | Convergence tolerance for optimization. |

