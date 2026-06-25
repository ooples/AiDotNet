---
title: "OrdinalLogisticRegression<T>"
description: "Ordinal Logistic Regression (Proportional Odds Model)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Classification.Ordinal`

Ordinal Logistic Regression (Proportional Odds Model).

## For Beginners

Ordinal logistic regression is the standard approach for ordinal data.
It's also known as the "proportional odds model" or "cumulative link model".

## How It Works

**How it works:** Instead of predicting the class directly, it models the cumulative
probability of being in class k or lower:

Where:

- **θₖ** = threshold (cutpoint) for class k
- **β** = coefficient vector (same for all classes - "proportional odds")
- **X** = feature vector

**Key assumption - Proportional Odds:** The effect of each feature is the same
regardless of which threshold we're considering. This means moving from 1→2 stars has
the same relationship with features as moving from 4→5 stars.

**Example:** Rating prediction for a restaurant review:

**References:**

- McCullagh, P. (1980). "Regression Models for Ordinal Data"

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OrdinalLogisticRegression(Double,Int32,Double,Double,Nullable<Int32>)` | Legacy constructor for backward compatibility with existing code and serialization. |
| `OrdinalLogisticRegression(IGradientBasedOptimizer<,Matrix<>,Vector<>>,Int32,Double,Double,Nullable<Int32>)` | Initializes a new instance of OrdinalLogisticRegression. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGradients(Vector<>,)` | Applies gradients to update model parameters. |
| `ComputeGradients(Matrix<>,Vector<>,ILossFunction<>)` | Computes gradients for the model parameters. |
| `CreateNewInstance` | Creates a new instance of this model type. |
| `Deserialize(Byte[])` |  |
| `GetFeatureImportance` | Gets feature importance based on coefficient magnitude. |
| `GetParameters` | Gets the model parameters (coefficients and thresholds). |
| `Predict(Matrix<>)` | Predicts ordinal class labels. |
| `PredictCumulativeProbabilities(Matrix<>)` | Predicts cumulative probabilities P(Y ≤ k). |
| `SanitizeParameters(Vector<>)` | Sanitizes random parameters to ensure ordinal constraints are satisfied. |
| `Serialize` |  |
| `SetParameters(Vector<>)` | Sets the model parameters. |
| `Train(Matrix<>,Vector<>)` | Gets the model type. |
| `WithParameters(Vector<>)` | Creates a new instance with the specified parameters. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_coefficients` | The learned coefficient vector (β). |
| `_learningRate` | Legacy learning rate field — kept for backward compatibility with serialization/clone. |
| `_maxIterations` | Maximum iterations for optimization. |
| `_paramOptimizer` | The gradient-based optimizer for parameter updates. |
| `_random` | Random number generator for initialization. |
| `_regularizationStrength` | L2 regularization strength. |
| `_tolerance` | Convergence tolerance. |

