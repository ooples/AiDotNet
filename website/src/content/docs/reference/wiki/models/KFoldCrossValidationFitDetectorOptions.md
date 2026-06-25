---
title: "KFoldCrossValidationFitDetectorOptions"
description: "Configuration options for the K-Fold Cross Validation Fit Detector, which evaluates model quality by analyzing performance across multiple data partitions."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the K-Fold Cross Validation Fit Detector, which evaluates model quality
by analyzing performance across multiple data partitions.

## For Beginners

K-Fold Cross Validation is like testing a recipe multiple times with
slightly different ingredients each time to make sure it's consistently good.

Instead of splitting your data just once into training and testing sets, K-Fold Cross Validation
divides your data into K equal parts (typically 5 or 10). It then runs K separate experiments:

- Experiment 1: Train on parts 2-K, test on part 1
- Experiment 2: Train on parts 1 and 3-K, test on part 2
- And so on...

This gives you K different performance measurements, which helps you understand:

- How consistent your model's performance is (does it work well on all parts of your data?)
- Whether your model is overfitting (working well on training data but poorly on test data)
- Whether your model is underfitting (working poorly on all data)

The Fit Detector analyzes these results automatically and tells you if there are any problems with
your model. This class lets you configure how sensitive the detector should be to different types
of problems.

## How It Works

K-Fold Cross Validation is a technique that divides the dataset into K equal parts (folds), then
trains and evaluates the model K times, each time using a different fold as the validation set and
the remaining folds as the training set. This provides a more robust assessment of model performance
than a single train-test split, especially for smaller datasets. The Fit Detector analyzes the
patterns in performance across these folds to identify overfitting, underfitting, and other model
quality issues.

## Properties

| Property | Summary |
|:-----|:--------|
| `GoodFitThreshold` | Gets or sets the threshold for confirming good fit based on performance and consistency across folds. |
| `HighVarianceThreshold` | Gets or sets the threshold for detecting high variance (inconsistent performance) across different folds. |
| `OverfitThreshold` | Gets or sets the threshold for detecting overfitting based on the difference between training and validation performance across folds. |
| `StabilityThreshold` | Gets or sets the threshold for assessing model stability based on performance consistency across multiple cross-validation runs. |
| `UnderfitThreshold` | Gets or sets the threshold for detecting underfitting based on the absolute performance level across folds. |

