---
title: "TLearner<T>"
description: "Implements the T-Learner (Two-model learner) for treatment effect estimation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalInference`

Implements the T-Learner (Two-model learner) for treatment effect estimation.

## For Beginners

T-Learner trains two separate models: one for the treated group
and one for the control group. Treatment effects are estimated by subtracting the control
model prediction from the treatment model prediction.

## How It Works

**How it works:**

- Train μ₁(X) on treated samples: {(Xᵢ, Yᵢ) : Tᵢ = 1}
- Train μ₀(X) on control samples: {(Xᵢ, Yᵢ) : Tᵢ = 0}
- Estimate CATE: τ(X) = μ₁(X) - μ₀(X)

**Pros and Cons:**

- **Pro:** Can capture complex heterogeneous treatment effects
- **Pro:** Each model is trained only on relevant data
- **Con:** Requires sufficient data in both groups
- **Con:** May have high variance when groups have different covariate distributions

**When to use:**

- When you have enough data in both treatment groups
- When treatment effects are expected to be heterogeneous
- When covariate distributions are similar across groups

**Reference:** Künzel et al., "Metalearners for Estimating Heterogeneous Treatment Effects" (2019)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TLearner(Int32,Double,Double)` | Creates a new T-Learner. |

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
| `EstimateTreatmentEffect(Matrix<>)` | Estimates the Conditional Average Treatment Effect (CATE). |
| `Fit(Matrix<>,Vector<>,Vector<>)` | Fits the T-Learner model. |
| `GetParameters` |  |
| `Predict(Matrix<>)` | Standard prediction - returns treatment effect. |
| `PredictControl(Matrix<>)` | Predicts outcome under control. |
| `PredictTreated(Matrix<>)` | Predicts outcome under treatment. |
| `PredictTreatmentEffect(Matrix<>)` |  |
| `SetParameters(Vector<>)` |  |
| `TrainLinearModel(Matrix<>,Vector<>,Int32[])` | Trains a linear regression model on specified indices. |
| `WithParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_biasControl` | Bias for the control model. |
| `_biasTreated` | Bias for the treatment model. |
| `_weightsControl` | Weights for the control model. |
| `_weightsTreated` | Weights for the treatment model. |

