---
title: "OrdinalRegression<T>"
description: "Implements Ordinal Regression (Proportional Odds Model) for predicting ordered categorical outcomes."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Classification`

Implements Ordinal Regression (Proportional Odds Model) for predicting ordered categorical outcomes.

## How It Works

Ordinal Regression is used when the target variable has naturally ordered categories. It models
the cumulative probability of being in category k or lower using the proportional odds assumption:
P(Y ≤ k) = sigmoid(α_k - β^T × x)
where α_k are ordered thresholds (one less than the number of classes) and β are feature coefficients.

Key properties:

- Respects the natural ordering of categories
- Uses single set of feature coefficients (proportional odds assumption)
- Ordered thresholds separate adjacent categories
- Probabilities for individual classes: P(Y = k) = P(Y ≤ k) - P(Y ≤ k-1)

For Beginners:
Ordinal Regression is perfect when your categories have a natural order but the distances
between them may not be equal. Examples include:

- Star ratings (1-5 stars): You know 5 > 4 > 3 > 2 > 1, but the difference between 1 and 2

stars might not equal the difference between 4 and 5 stars

- Survey responses: Strongly Disagree < Disagree < Neutral < Agree < Strongly Agree
- Education levels: High School < Bachelor's < Master's < PhD
- Pain levels: None < Mild < Moderate < Severe

The model learns:

1. Feature coefficients (β): How each feature pushes predictions up or down the ordinal scale
2. Thresholds (α): Where to draw the lines between adjacent categories

This is better than treating it as regular classification (which ignores order) or as
regression (which assumes equal distances between categories).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OrdinalRegression(OrdinalRegressionOptions<>,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of the OrdinalRegression class with the specified options and regularization. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Coefficients` | Gets the feature coefficients. |
| `Thresholds` | Gets the threshold parameters. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGradients(Vector<>,)` | Applies gradients to update the model parameters. |
| `ApplyLink(Double)` | Applies the link function to convert linear predictor to cumulative probability. |
| `Clone` |  |
| `ComputeClassProbabilities(Matrix<>,Int32)` | Computes the probability distribution over classes for a single sample. |
| `ComputeGradients(Matrix<>,Vector<>)` | Computes gradients for the proportional odds model. |
| `ComputeGradients(Matrix<>,Vector<>,ILossFunction<>)` | Computes gradients for the model parameters. |
| `CreateNewInstance` | Creates a new instance of the same type as this classifier. |
| `DeepCopy` |  |
| `EnforceThresholdOrdering` | Enforces the ordering constraint on thresholds: α_1 < α_2 < ... |
| `Erf(Double)` | Computes the error function approximation. |
| `GetFeatureImportance` | Gets the feature importance scores (absolute coefficient values). |
| `GetOptions` |  |
| `GetParameters` | Gets all model parameters as a single vector (coefficients + thresholds). |
| `NormalCdf(Double)` | Computes the standard normal cumulative distribution function. |
| `Predict(Matrix<>)` | Predicts class labels for the given input data. |
| `PredictProbabilities(Matrix<>)` | Returns the probability estimates for all classes. |
| `SetParameters(Vector<>)` | Sets the model parameters from a vector. |
| `Train(Matrix<>,Vector<>)` | Trains the ordinal regression model on the provided data. |
| `ValidateOrdinalData(Vector<>)` | Validates that the target values are valid ordinal classes (integers 0 to K-1). |
| `WithParameters(Vector<>)` | Creates a new instance of the model with specified parameters. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_coefficients` | Feature coefficients (β). |
| `_options` | Configuration options for the ordinal regression model. |

