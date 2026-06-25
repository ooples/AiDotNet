---
title: "OnlineSGDRegressor<T>"
description: "Online Stochastic Gradient Descent regressor for incremental learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.OnlineLearning`

Online Stochastic Gradient Descent regressor for incremental learning.

## For Beginners

This regressor predicts continuous values using a linear model,
updating itself one example at a time.

How it works:

1. Compute prediction: ŷ = w·x + b
2. Compare with true value: error = prediction - truth
3. Update weights: w = w - learning_rate × error × x

Over many examples, the model converges to the best linear fit.

Supports multiple loss functions:

- Squared error (default): Sensitive to outliers
- Huber loss: Robust to outliers
- Epsilon-insensitive: SVR-like, ignores errors smaller than epsilon

Usage:

References:

- Bottou, L. (2010). "Large-Scale Machine Learning with Stochastic Gradient Descent"

## How It Works

Online SGD Regressor implements linear regression with SGD updates, allowing
the model to learn incrementally from streaming data.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OnlineSGDRegressor(Double,LearningRateSchedule,Double,Double,Boolean,SGDLossType,Double)` | Gets the model type. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeEpsilonInsensitiveGradient(Double)` | Epsilon-insensitive loss gradient: zero for small errors. |
| `ComputeHuberGradient(Double)` | Huber loss gradient: linear for large errors, quadratic for small. |
| `ComputeLossGradient(Double,Double)` | Computes the gradient of the loss function. |
| `ComputePrediction(Vector<>)` | Computes the prediction for a sample. |
| `CreateNewInstance` | Creates a new instance of this type. |
| `GetBias` | Gets the bias (intercept) term. |
| `GetFeatureImportance` | Gets the feature importance scores (absolute weights). |
| `GetParameters` | Gets the model parameters (weights + bias). |
| `GetWeights` | Gets the weights vector. |
| `PartialFit(Vector<>,)` | Updates the model with a single training example. |
| `PredictSingle(Vector<>)` | Predicts the target value for a single sample. |
| `Reset` | Resets the model to its initial state. |
| `Score(Matrix<>,Vector<>)` | Gets the R-squared score on the provided data. |
| `SetParameters(Vector<>)` | Sets the model parameters. |
| `SoftThreshold(Double,Double)` | Soft thresholding operator for L1 regularization. |
| `WithParameters(Vector<>)` | Creates a new instance with specified parameters. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_bias` | The bias (intercept) term. |
| `_epsilon` | Epsilon for epsilon-insensitive and Huber loss. |
| `_fitIntercept` | Whether to fit an intercept (bias) term. |
| `_l1Penalty` | L1 regularization strength. |
| `_l2Penalty` | L2 regularization strength. |
| `_lossType` | The loss function type. |
| `_weights` | The weight vector (coefficients). |

