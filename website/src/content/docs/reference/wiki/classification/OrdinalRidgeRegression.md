---
title: "OrdinalRidgeRegression<T>"
description: "Ordinal Ridge Regression using the Immediate-Threshold approach with L2 regularization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Classification.Ordinal`

Ordinal Ridge Regression using the Immediate-Threshold approach with L2 regularization.

## For Beginners

This is a variant of ordinal classification that treats the problem
as a regression task with specially-designed thresholds. It uses ridge (L2) regularization
to prevent overfitting, which adds a penalty for large coefficient values.

## How It Works

**How it works:** Instead of modeling cumulative probabilities like ordinal logistic
regression, this approach:

- Treats the ordinal labels as numeric targets
- Learns a linear function f(X) = β·X
- Uses thresholds to convert predictions back to ordinal classes

**Immediate-Threshold method:** After training the regression model, thresholds are
placed at the midpoints between consecutive class means in the training data. This gives
natural boundaries between classes.

**Ridge regularization:** Adds λ·||β||² to the loss function, which:

- Prevents coefficients from becoming too large
- Reduces overfitting on noisy data
- Provides a closed-form solution (no iterative optimization needed)

**When to use:**

- When you want a fast, closed-form solution
- When the ordinal levels are roughly equally spaced
- When you need regularization to handle multicollinearity

**References:**

- Frank, E. & Hall, M. (2001). "A Simple Approach to Ordinal Classification"
- Chu, W. & Keerthi, S.S. (2007). "Support Vector Ordinal Regression"

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OrdinalRidgeRegression(Double,Boolean)` | Initializes a new instance of OrdinalRidgeRegression. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGradients(Vector<>,)` | Applies gradients to update model parameters. |
| `ComputeGradients(Matrix<>,Vector<>,ILossFunction<>)` | Computes gradients for the model parameters. |
| `CreateNewInstance` | Creates a new instance of this model type. |
| `Deserialize(Byte[])` |  |
| `GetFeatureImportance` | Gets feature importance based on coefficient magnitude. |
| `GetParameters` | Gets the model parameters (coefficients, bias, and thresholds). |
| `Predict(Matrix<>)` | Predicts ordinal class labels. |
| `PredictCumulativeProbabilities(Matrix<>)` | Predicts cumulative probabilities P(Y ≤ k). |
| `Serialize` |  |
| `SetParameters(Vector<>)` | Sets the model parameters. |
| `Train(Matrix<>,Vector<>)` | Gets the model type. |
| `WithParameters(Vector<>)` | Creates a new instance with the specified parameters. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_alpha` | Ridge regularization parameter (λ). |
| `_bias` | The bias (intercept) term. |
| `_coefficients` | The learned coefficient vector (β). |
| `_fitIntercept` | Whether to fit an intercept term. |

