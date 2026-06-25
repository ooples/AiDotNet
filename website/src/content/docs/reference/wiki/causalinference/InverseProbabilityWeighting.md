---
title: "InverseProbabilityWeighting<T>"
description: "Implements Inverse Probability Weighting (IPW) for causal effect estimation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalInference`

Implements Inverse Probability Weighting (IPW) for causal effect estimation.

## For Beginners

IPW works by giving more weight to "surprising" observations:

- A treated person who was unlikely to be treated gets high weight
- A control person who was likely to be treated gets high weight

Why? In observational data, treatment isn't random - some people are more likely
to be treated. IPW corrects for this by "up-weighting" under-represented cases.

Example - Job Training Study:

- Highly motivated person who didn't get training: weight = 1/(1-0.9) = 10
- Less motivated person who did get training: weight = 1/0.2 = 5
- This gives more influence to the "surprising" cases

Mathematical formula:
ATE = E[T×Y/e(X)] - E[(1-T)×Y/(1-e(X))]

Where:

- T = treatment (0 or 1)
- Y = outcome
- e(X) = propensity score = P(T=1|X)

Advantages:

- Uses all data (unlike matching which discards unmatched)
- Computationally simple
- Easy to combine with regression (augmented IPW)

Limitations:

- Can be unstable if propensity scores are extreme (near 0 or 1)
- Sensitive to propensity score model misspecification

References:

- Horvitz & Thompson (1952). "A Generalization of Sampling Without Replacement"
- Robins, Rotnitzky & Zhao (1994). "Estimation of Regression Coefficients"

## How It Works

IPW estimates treatment effects by weighting observations inversely by their probability
of receiving their actual treatment status, creating a pseudo-population where treatment
is independent of confounders.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `InverseProbabilityWeighting(Double,Double,Boolean)` | Gets the model type. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeWeights(Matrix<>,Vector<Int32>)` | Computes IPW weights for each observation. |
| `CreateNewInstance` | Creates a new instance of this type. |
| `EstimateATE(Matrix<>,Vector<Int32>,Vector<>)` | Estimates the Average Treatment Effect using IPW. |
| `EstimateATT(Matrix<>,Vector<Int32>,Vector<>)` | Estimates the Average Treatment Effect on the Treated using IPW. |
| `EstimateCATEPerIndividual(Matrix<>,Vector<Int32>,Vector<>)` | Estimates individual treatment effects. |
| `EstimatePropensityScoresCore(Matrix<>)` | Estimates propensity scores using the fitted model. |
| `EstimateTreatmentEffect(Matrix<>)` | Estimates treatment effects for individuals using IPW. |
| `Fit(Matrix<>,Vector<>,Vector<>)` | Fits the causal model using the ICausalModel interface signature. |
| `Fit(Matrix<>,Vector<Int32>)` | Fits the propensity score model to the data. |
| `GetAdditionalModelData` | Gets additional model data for serialization. |
| `GetEffectiveSampleSize(Matrix<>,Vector<Int32>)` | Gets the effective sample size after weighting. |
| `GetParameters` | Gets the model parameters (propensity score coefficients). |
| `LoadAdditionalModelData(JObject)` | Loads additional model data from deserialization. |
| `Predict(Matrix<>)` | Standard prediction - returns propensity scores. |
| `PredictControl(Matrix<>)` | Predicts outcomes under control for the given features. |
| `PredictTreated(Matrix<>)` | Predicts outcomes under treatment for the given features. |
| `PredictTreatmentEffect(Matrix<>)` | Predicts treatment effects for new individuals. |
| `SetParameters(Vector<>)` | Sets the model parameters. |
| `WithParameters(Vector<>)` | Creates a new instance with specified parameters. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_cachedFeatures` | Cached feature matrix from fitting. |
| `_cachedOutcome` | Cached outcome vector from fitting. |
| `_cachedTreatment` | Cached treatment vector from fitting. |
| `_propensityCoefficients` | Stores the logistic regression coefficients for propensity score estimation. |
| `_stabilizedWeights` | Whether to use stabilized weights. |
| `_trimMax` | Maximum propensity score to avoid extreme weights. |
| `_trimMin` | Minimum propensity score to avoid extreme weights. |

