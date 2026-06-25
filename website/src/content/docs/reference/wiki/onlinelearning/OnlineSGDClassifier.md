---
title: "OnlineSGDClassifier<T>"
description: "Online Stochastic Gradient Descent classifier for incremental learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.OnlineLearning`

Online Stochastic Gradient Descent classifier for incremental learning.

## For Beginners

This classifier learns to separate two classes using a linear
decision boundary, updating itself one example at a time.

How it works:

1. Compute prediction: P(y=1) = sigmoid(w·x + b)
2. Compare with true label: error = prediction - truth
3. Update weights: w = w - learning_rate × error × x

The model "nudges" itself toward correct predictions with each example.
Over many examples, it converges to a good decision boundary.

Advantages:

- Handles streaming data naturally
- Memory-efficient (doesn't store data)
- Can adapt to changing patterns

Supports:

- L1 regularization (sparse weights)
- L2 regularization (smooth weights)
- Elastic Net (combination)

Usage:

References:

- Bottou, L. (2010). "Large-Scale Machine Learning with Stochastic Gradient Descent"

## How It Works

Online SGD Classifier implements logistic regression with SGD updates, allowing
the model to learn incrementally from streaming data.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OnlineSGDClassifier(Double,LearningRateSchedule,Double,Double,Boolean)` | Gets the model type. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `ComputeProbability(Vector<>)` | Computes the probability of class 1. |
| `CreateNewInstance` | Creates a new instance of this type. |
| `DecisionFunction(Vector<>)` | Computes the score used for decision (before sigmoid). |
| `GetBias` | Gets the bias (intercept) term. |
| `GetFeatureImportance` | Gets the feature importance scores (absolute weights). |
| `GetParameters` | Gets the model parameters (weights + bias). |
| `GetWeights` | Gets the weights vector. |
| `PartialFit(Vector<>,)` | Updates the model with a single training example. |
| `PredictProbabilities(Matrix<>)` | Predicts probabilities for all samples. |
| `PredictProbability(Vector<>)` | Predicts the probability of class 1. |
| `PredictSingle(Vector<>)` | Predicts the target value for a single sample. |
| `Reset` | Resets the model to its initial state. |
| `SetParameters(Vector<>)` | Sets the model parameters. |
| `Sigmoid(Double)` | Sigmoid function. |
| `SoftThreshold(Double,Double)` | Soft thresholding operator for L1 regularization. |
| `WithParameters(Vector<>)` | Creates a new instance with specified parameters. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_bias` | The bias (intercept) term. |
| `_fitIntercept` | Whether to fit an intercept (bias) term. |
| `_l1Penalty` | L1 regularization strength. |
| `_l2Penalty` | L2 regularization strength. |
| `_weights` | The weight vector (coefficients). |

