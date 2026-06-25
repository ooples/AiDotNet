---
title: "ExplainableBoostingClassifier<T>"
description: "Explainable Boosting Machine (EBM) for interpretable classification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Classification.Boosting`

Explainable Boosting Machine (EBM) for interpretable classification.

## For Beginners

EBM is special because it gives you the best of both worlds:

- High accuracy (comparable to gradient boosting and random forests)
- Full interpretability (you can see exactly why each prediction was made)

How it works:

1. For each feature, EBM learns a "shape function" that shows how that feature

affects the prediction

2. The final prediction is simply the sum of all these shape functions plus

an intercept, passed through sigmoid for probability

3. You can plot these shape functions to understand exactly how the model

uses each feature

For example, in predicting loan defaults:

- The shape function for "income" might show higher income = lower risk
- The shape function for "debt_ratio" might show higher ratio = higher risk
- The prediction combines: intercept + f(income) + f(debt_ratio) + ...

This additive structure makes EBM uniquely interpretable while still being accurate.

## How It Works

EBM is a Generalized Additive Model (GAM) with boosting that provides glass-box
interpretability while maintaining high accuracy. It learns smooth functions for
each feature and optionally pairwise interactions.

Reference: Lou, Y., et al. "Intelligible Models for Healthcare: Predicting
Pneumonia Risk and Hospital 30-day Readmission" (2012).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ExplainableBoostingClassifier(ExplainableBoostingClassifierOptions<>,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of ExplainableBoostingClassifier. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BinEdges` | Gets the bin edges for each feature. |
| `InteractionTerms` | Gets the interaction terms. |
| `Intercept` | Gets the intercept (baseline log-odds). |
| `ShapeFunctions` | Gets the shape functions for each feature. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateFeatureImportances` | Calculates feature importances based on shape function variability. |
| `CenterShapeFunctions(Int32[][],Int32)` | Centers shape functions by subtracting their weighted mean. |
| `ComputeInteractionScore(Int32[],Int32[],Vector<>)` | Computes interaction score (variance reduction). |
| `ComputeLogLoss(Vector<>,Vector<>)` | Computes log loss. |
| `ComputeLogOdds(Matrix<>,Int32[][])` | Computes log-odds for all samples. |
| `ComputeVariance(Vector<>)` | Computes variance. |
| `CreateBins(Matrix<>)` | Creates bins for each feature using quantiles. |
| `CreateNewInstance` |  |
| `Deserialize(Byte[])` |  |
| `DetectInteractions(Matrix<>,Int32[][],Vector<>)` | Detects important pairwise interactions and trains them with cyclic boosting. |
| `ExplainPrediction(Vector<>)` | Explains a single prediction by showing each feature's contribution. |
| `GetBinIndex(,Int32)` | Gets the bin index for a value. |
| `GetModelMetadata` |  |
| `Predict(Matrix<>)` | Predicts class labels for input samples. |
| `PredictProbabilities(Matrix<>)` | Predicts class probabilities for input samples. |
| `Serialize` |  |
| `Sigmoid()` | Sigmoid function. |
| `Train(Matrix<>,Vector<>)` | Returns the model type identifier. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_binEdges` | Bin edges for each feature. |
| `_interactionTerms` | Interaction terms: pairs of features and their joint effect. |
| `_intercept` | The intercept (baseline log-odds). |
| `_numFeatures` | Number of features. |
| `_options` | Configuration options. |
| `_random` | Random number generator. |
| `_shapeFunctions` | Shape functions for each feature (additive terms). |

