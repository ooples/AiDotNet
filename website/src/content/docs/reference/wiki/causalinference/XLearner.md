---
title: "XLearner<T>"
description: "Implements the X-Learner (Cross-learner) for treatment effect estimation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalInference`

Implements the X-Learner (Cross-learner) for treatment effect estimation.

## For Beginners

X-Learner is a sophisticated meta-learner that adapts to the data
by using cross-fitting. It's especially effective when treatment and control groups have
different sizes, as it leverages information from both groups more efficiently.

## How It Works

**How it works (5 stages):**

- Train μ₀(X) and μ₁(X) using T-Learner approach
- Impute treatment effects: D₁ᵢ = Y₁ᵢ - μ₀(X₁ᵢ) for treated, D₀ᵢ = μ₁(X₀ᵢ) - Y₀ᵢ for control
- Train τ₁(X) on D₁ (treated imputed effects) and τ₀(X) on D₀ (control imputed effects)
- Estimate propensity score e(X) = P(T=1|X)
- Combine: τ(X) = e(X)·τ₀(X) + (1-e(X))·τ₁(X)

**Key Insight:** The weighted combination uses propensity scores to give more weight
to the model trained on the larger group, making X-Learner robust to imbalanced data.

**Pros and Cons:**

- **Pro:** Excellent for imbalanced treatment groups
- **Pro:** Can outperform T-Learner when one group is much smaller
- **Pro:** Adapts to the data structure through propensity weighting
- **Con:** More complex, requires fitting 5 models
- **Con:** Propensity estimation can be sensitive

**When to use:**

- When treatment/control groups are imbalanced
- When you want state-of-the-art CATE estimation
- When you have sufficient data for multiple model fitting

**Reference:** Künzel et al., "Metalearners for Estimating Heterogeneous Treatment Effects" (2019)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `XLearner(Int32,Double,Double)` | Creates a new X-Learner. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Lambda` | Gets the L2 regularization strength. |
| `LearningRate` | Gets the learning rate for training. |
| `MaxIterations` | Gets the maximum iterations for training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |
| `EstimateATE(Matrix<>,Vector<Int32>,Vector<>)` |  |
| `EstimateATT(Matrix<>,Vector<Int32>,Vector<>)` |  |
| `EstimateCATEPerIndividual(Matrix<>,Vector<Int32>,Vector<>)` |  |
| `EstimatePropensityScoresCore(Matrix<>)` |  |
| `EstimateTreatmentEffect(Matrix<>)` | Estimates the Conditional Average Treatment Effect (CATE) using propensity-weighted combination. |
| `Fit(Matrix<>,Vector<>,Vector<>)` | Fits the X-Learner model using the 5-stage algorithm. |
| `GetAdditionalModelData` |  |
| `GetParameters` |  |
| `LoadAdditionalModelData(JObject)` |  |
| `Predict(Matrix<>)` | Standard prediction - returns treatment effect. |
| `PredictControl(Matrix<>)` | Predicts outcome under control using μ₀(X). |
| `PredictSingle(Matrix<>,Int32,Vector<>,)` | Predicts a single outcome using given weights. |
| `PredictTreated(Matrix<>)` | Predicts outcome under treatment using μ₁(X). |
| `PredictTreatmentEffect(Matrix<>)` |  |
| `SetParameters(Vector<>)` |  |
| `TrainLinearModel(Matrix<>,Vector<>,Int32[])` | Trains a linear regression model on specified indices. |
| `TrainLinearModelWithOutcome(Matrix<>,Vector<>,Int32[])` | Trains a linear model with a separate outcome vector (for imputed effects). |
| `TrainPropensityModel(Matrix<>,Vector<Int32>)` | Trains a logistic regression model for propensity scores. |
| `WithParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_biasControl` | Bias terms for each model. |
| `_biasPropensity` | Bias terms for each model. |
| `_biasTau0` | Bias terms for each model. |
| `_biasTau1` | Bias terms for each model. |
| `_biasTreated` | Bias terms for each model. |
| `_weightsControl` | Weights for the control outcome model μ₀. |
| `_weightsPropensity` | Weights for the propensity score model. |
| `_weightsTau0` | Weights for the treatment effect model τ₀ (trained on control imputed effects). |
| `_weightsTau1` | Weights for the treatment effect model τ₁ (trained on treated imputed effects). |
| `_weightsTreated` | Weights for the treatment outcome model μ₁. |

