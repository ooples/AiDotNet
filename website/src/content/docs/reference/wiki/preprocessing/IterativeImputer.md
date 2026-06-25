---
title: "IterativeImputer<T>"
description: "Iterative imputer using the MICE algorithm (Multiple Imputation by Chained Equations)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.Imputers`

Iterative imputer using the MICE algorithm (Multiple Imputation by Chained Equations).

## For Beginners

MICE creates multiple "guesses" for missing values by
learning relationships between features. If taller people tend to be heavier,
MICE can use height to predict missing weight values more accurately than
simply using the average weight.

## How It Works

IterativeImputer imputes missing values by modeling each feature with missing values
as a function of other features, iterating multiple times until convergence.

The algorithm:

1. Initial imputation (mean/median for each feature)
2. For each feature with missing values:
- Train a regression model using other features as predictors
- Predict missing values using the trained model
3. Repeat step 2 for multiple iterations until convergence

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `IterativeImputer(Int32,Double,IterativeImputerEstimator,IterativeImputerInitialStrategy,Int32,Int32[])` | Creates a new instance of `IterativeImputer`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Estimator` | Gets the estimator type used for imputation. |
| `InitialStrategy` | Gets the initial imputation strategy. |
| `MaxIterations` | Gets the maximum number of iterations. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |
| `Tolerance` | Gets the convergence tolerance. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitCore(Matrix<>)` | Fits the imputer by learning the relationships between features. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported. |
| `TransformCore(Matrix<>)` | Transforms the data by imputing missing values. |

