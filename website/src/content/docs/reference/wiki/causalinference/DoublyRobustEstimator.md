---
title: "DoublyRobustEstimator<T>"
description: "Implements the Doubly Robust (DR) estimator for causal effect estimation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalInference`

Implements the Doubly Robust (DR) estimator for causal effect estimation.

## For Beginners

The DR estimator is like having two insurance policies:

1. Outcome Model: Predicts outcomes based on features and treatment
2. Propensity Model: Predicts who gets treated

If either model is correct, you get unbiased treatment effect estimates.
This "double protection" makes DR very popular in practice.

The formula combines:

- Predicted outcomes (if we trust the outcome model)
- IPW-adjusted residuals (correction if outcome model is wrong)

DR estimator:
τ̂ = (1/n) Σ [μ₁(Xᵢ) - μ₀(Xᵢ)] + (1/n) Σ [Tᵢ(Yᵢ - μ₁(Xᵢ))/e(Xᵢ) - (1-Tᵢ)(Yᵢ - μ₀(Xᵢ))/(1-e(Xᵢ))]

Where:

- μ₁(X), μ₀(X) = predicted outcomes under treatment/control
- e(X) = propensity score
- T = treatment indicator
- Y = observed outcome

The first term uses the outcome model predictions.
The second term "corrects" using IPW when the outcome model is wrong.

Advantages:

- Doubly robust: consistent if either model is correct
- More efficient than IPW alone when both models are good
- Semiparametric efficiency (achieves best possible variance)

References:

- Robins, Rotnitzky & Zhao (1994). "Estimation of Regression Coefficients"
- Bang & Robins (2005). "Doubly Robust Estimation"

## How It Works

The Doubly Robust estimator combines outcome regression with propensity score weighting,
providing consistent estimates if EITHER the outcome model OR the propensity model is correct.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DoublyRobustEstimator(Double,Double,Boolean,Int32)` | Initializes a new instance of the DoublyRobustEstimator class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Creates a new instance of this type. |
| `EstimateATE(Matrix<>,Vector<Int32>,Vector<>)` | Estimates the Average Treatment Effect using the Doubly Robust estimator. |
| `EstimateATT(Matrix<>,Vector<Int32>,Vector<>)` | Estimates the Average Treatment Effect on the Treated. |
| `EstimateCATEPerIndividual(Matrix<>,Vector<Int32>,Vector<>)` | Estimates individual treatment effects. |
| `EstimatePropensityScoresCore(Matrix<>)` | Estimates propensity scores using the fitted model. |
| `EstimateTreatmentEffect(Matrix<>)` | Estimates treatment effects for individuals using the doubly robust estimator. |
| `Fit(Matrix<>,Vector<>,Vector<>)` | Fits the causal model using the ICausalModel interface signature. |
| `Fit(Matrix<>,Vector<Int32>,Vector<>)` | Fits both propensity score and outcome models to the data. |
| `FitLinearRegressionOLS(List<Double[]>,List<Double>,Int32)` | Simple OLS linear regression fitting. |
| `FitOutcomeModels(Matrix<>,Vector<Int32>,Vector<>)` | Fits separate linear regression models for treated and control groups. |
| `GetParameters` | Gets the model parameters (propensity + outcome coefficients). |
| `Predict(Matrix<>)` | Standard prediction - returns predicted treatment effects. |
| `PredictControl(Matrix<>)` | Predicts outcomes under control for the given features. |
| `PredictOutcome(Matrix<>,Vector<>)` | Predicts outcomes using the outcome regression model. |
| `PredictTreated(Matrix<>)` | Predicts outcomes under treatment for the given features. |
| `PredictTreatmentEffect(Matrix<>)` | Predicts treatment effects for new individuals. |
| `SetParameters(Vector<>)` | Sets the model parameters. |
| `SolveLinearSystem(Double[0:,0:],Double[],Int32)` | Solves a linear system using Gaussian elimination with partial pivoting. |
| `WithParameters(Vector<>)` | Creates a new instance with specified parameters. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_numFolds` | Number of folds for cross-fitting. |
| `_outcomeCoefficients0` | Stores the outcome regression coefficients for control group. |
| `_outcomeCoefficients1` | Stores the outcome regression coefficients for treated group. |
| `_propensityCoefficients` | Stores the logistic regression coefficients for propensity score estimation. |
| `_trimMax` | Maximum propensity score to avoid extreme weights. |
| `_trimMin` | Minimum propensity score to avoid extreme weights. |
| `_useCrossFitting` | Whether to use cross-fitting for debiased estimation. |

