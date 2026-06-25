---
title: "SLearner<T>"
description: "Implements the S-Learner (Single-model learner) for treatment effect estimation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalInference`

Implements the S-Learner (Single-model learner) for treatment effect estimation.

## For Beginners

S-Learner is the simplest meta-learner. It trains a single model
that predicts outcomes using both covariates AND the treatment indicator as features.
Treatment effects are estimated by comparing predictions with treatment=1 vs treatment=0.

## How It Works

**How it works:**

- Train a single model: Y = f(X, T) where T is the treatment indicator
- For each subject, predict Y₁ = f(X, T=1) and Y₀ = f(X, T=0)
- Treatment effect τ(X) = Y₁ - Y₀

**Pros and Cons:**

- **Pro:** Simple, uses all data efficiently
- **Pro:** Works with any supervised learning method
- **Con:** May underestimate heterogeneous treatment effects if treatment has small signal
- **Con:** Regularization may shrink treatment effect toward zero

**When to use:**

- When you have limited data and can't afford separate models
- When treatment effects are expected to be relatively homogeneous
- As a baseline to compare against more complex learners

**Reference:** Künzel et al., "Metalearners for Estimating Heterogeneous Treatment Effects" (2019)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SLearner(Int32,Double,Double)` | Creates a new S-Learner. |

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
| `Fit(Matrix<>,Vector<>,Vector<>)` | Fits the S-Learner model. |
| `GetAdditionalModelData` |  |
| `GetParameters` |  |
| `LoadAdditionalModelData(JObject)` |  |
| `Predict(Matrix<>)` | Standard prediction — returns the predicted outcome for each input row. |
| `PredictControl(Matrix<>)` | Predicts outcome under control. |
| `PredictTreated(Matrix<>)` | Predicts outcome under treatment. |
| `PredictTreatmentEffect(Matrix<>)` |  |
| `SetParameters(Vector<>)` |  |
| `WithParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_bias` | The bias term. |
| `_weights` | The model weights (including treatment as a feature). |

