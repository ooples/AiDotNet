---
title: "OnlinePassiveAggressiveRegressor<T>"
description: "Online Passive-Aggressive regressor for epsilon-insensitive incremental regression."
section: "API Reference"
---

`Models & Types` · `AiDotNet.OnlineLearning`

Online Passive-Aggressive regressor for epsilon-insensitive incremental regression.

## For Beginners

PA regressors work like strict teachers for continuous predictions:

- Passive: When the prediction error is within ε → do nothing
- Aggressive: When error exceeds ε → update strongly to reduce the error

How it works:

1. Compute prediction: ŷ = w·x + b
2. Compute error: e = |y - ŷ|
3. If e ≤ ε (epsilon): Acceptable error → stay passive
4. If e > ε: Too much error → aggressively update weights

The ε (epsilon) parameter defines an "acceptable error zone" similar to
epsilon-insensitive loss in SVR.

PA variants:

- PA: Original, no regularization (can diverge with noise)
- PA-I: Adds slack variable, bounds the update size
- PA-II: Adds squared penalty, smoother updates

Advantages over standard online regression:

- No learning rate to tune (automatically determined)
- Robust to small errors (epsilon-insensitive)
- Fast convergence on well-behaved data

Usage:

References:

- Crammer et al. (2006). "Online Passive-Aggressive Algorithms"

## How It Works

Passive-Aggressive (PA) regression algorithms are a family of online learning algorithms that
aggressively update when the prediction error exceeds a threshold but remain passive when
the error is within tolerance.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OnlinePassiveAggressiveRegressor(Double,Double,PAType,Boolean)` | Gets the model type. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Epsilon` | Gets the epsilon parameter (insensitivity zone). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputePrediction(Vector<>)` | Computes the prediction (before epsilon zone). |
| `ComputeStepSize(Double,Double)` | Computes the step size based on PA variant. |
| `CreateNewInstance` | Creates a new instance of this type. |
| `GetBias` | Gets the bias (intercept) term. |
| `GetEpsilonInsensitiveLoss(Matrix<>,Vector<>)` | Computes the epsilon-insensitive loss on the provided data. |
| `GetFeatureImportance` | Gets the feature importance scores (absolute weights). |
| `GetMeanSquaredError(Matrix<>,Vector<>)` | Computes the mean squared error on the provided data. |
| `GetParameters` | Gets the model parameters (weights + bias). |
| `GetWeights` | Gets the weights vector. |
| `PartialFit(Vector<>,)` | Updates the model with a single training example. |
| `PredictSingle(Vector<>)` | Predicts the value for a single sample. |
| `Reset` | Resets the model to its initial state. |
| `SetParameters(Vector<>)` | Sets the model parameters. |
| `WithParameters(Vector<>)` | Creates a new instance with specified parameters. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_bias` | The bias (intercept) term. |
| `_c` | Regularization parameter (aggressiveness). |
| `_epsilon` | Epsilon parameter for insensitivity zone. |
| `_fitIntercept` | Whether to fit an intercept (bias) term. |
| `_paType` | PA variant type. |
| `_weights` | The weight vector (coefficients). |

