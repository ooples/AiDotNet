---
title: "StratifiedKFoldCrossValidationFitDetectorOptions"
description: "Configuration options for detecting overfitting, underfitting, and model stability using stratified k-fold cross-validation."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for detecting overfitting, underfitting, and model stability using
stratified k-fold cross-validation.

## For Beginners

This class helps you detect common model training problems using cross-validation results.

When training machine learning models:

- Overfitting: Model learns the training data too well but doesn't generalize
- Underfitting: Model is too simple and doesn't capture important patterns
- High variance: Model performance changes dramatically with different data subsets

Stratified k-fold cross-validation:

- Splits your data into k subsets (folds)
- Maintains the same class distribution in each fold (stratified)
- Trains k different models, each using k-1 folds for training and 1 for validation
- Helps assess how well your model will generalize to new data

This class provides thresholds to automatically detect these issues based on
cross-validation results, helping you diagnose and fix model training problems.

## How It Works

Stratified K-Fold Cross-Validation is a technique that divides the dataset into k folds (subsets) while 
maintaining the same class distribution in each fold as in the complete dataset. This is particularly 
important for imbalanced datasets where some classes have significantly fewer samples than others. The 
fit detector uses the performance metrics across these folds to assess whether a model is overfitting 
(performing much better on training data than validation data), underfitting (performing poorly on both 
training and validation data), or has high variance (performance varies significantly across different 
folds). This class provides configuration options for the thresholds used to make these determinations.

## Properties

| Property | Summary |
|:-----|:--------|
| `GoodFitThreshold` | Gets or sets the threshold for determining a good fit. |
| `HighVarianceThreshold` | Gets or sets the threshold for detecting high variance. |
| `OverfitThreshold` | Gets or sets the threshold for detecting overfitting. |
| `PrimaryMetric` | Gets or sets the primary metric used for evaluating model fit. |
| `StabilityThreshold` | Gets or sets the threshold for determining model stability. |
| `UnderfitThreshold` | Gets or sets the threshold for detecting underfitting. |

