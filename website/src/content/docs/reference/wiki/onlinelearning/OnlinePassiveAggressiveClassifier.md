---
title: "OnlinePassiveAggressiveClassifier<T>"
description: "Online Passive-Aggressive classifier for margin-based incremental learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.OnlineLearning`

Online Passive-Aggressive classifier for margin-based incremental learning.

## For Beginners

PA classifiers are like strict teachers:

- Passive: When the prediction is correct with good margin → do nothing
- Aggressive: When wrong or uncertain → update strongly to fix the mistake

How it works:

1. Compute margin: y × (w·x) - how confident and correct is the prediction?
2. If margin >= 1: Correct with good margin → stay passive
3. If margin < 1: Wrong or uncertain → aggressively update

The update is designed to:

- Correct the mistake with minimum change to weights
- Maintain a margin of at least 1 after update

PA variants:

- PA: Original, no regularization (can diverge with noise)
- PA-I: Adds slack variable, bounds the update size
- PA-II: Adds squared penalty, smoother updates

Advantages over SGD:

- No learning rate to tune (automatically determined)
- Fast convergence on linearly separable data
- Naturally handles the margin

Usage:

References:

- Crammer et al. (2006). "Online Passive-Aggressive Algorithms"

## How It Works

Passive-Aggressive (PA) algorithms are a family of online learning algorithms that
aggressively update when a mistake is made but remain passive when the prediction is correct.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OnlinePassiveAggressiveClassifier(Double,PAType,Boolean)` | Gets the model type. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `ComputeScore(Vector<>)` | Computes the score (before sign function). |
| `ComputeStepSize(Double,Double)` | Computes the step size based on PA variant. |
| `CreateNewInstance` | Creates a new instance of this type. |
| `DecisionFunction(Vector<>)` | Gets the decision function value (raw score before sign). |
| `GetBias` | Gets the bias (intercept) term. |
| `GetFeatureImportance` | Gets the feature importance scores (absolute weights). |
| `GetHingeLoss(Matrix<>,Vector<>)` | Computes the hinge loss on the provided data. |
| `GetParameters` | Gets the model parameters (weights + bias). |
| `GetWeights` | Gets the weights vector. |
| `PartialFit(Vector<>,)` | Updates the model with a single training example. |
| `PredictBinary(Vector<>)` | Converts predictions to 0/1 format for compatibility. |
| `PredictSingle(Vector<>)` | Predicts the class label for a single sample. |
| `Reset` | Resets the model to its initial state. |
| `SetParameters(Vector<>)` | Sets the model parameters. |
| `WithParameters(Vector<>)` | Creates a new instance with specified parameters. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_bias` | The bias (intercept) term. |
| `_c` | Regularization parameter (aggressiveness). |
| `_fitIntercept` | Whether to fit an intercept (bias) term. |
| `_paType` | PA variant type. |
| `_weights` | The weight vector (coefficients). |

