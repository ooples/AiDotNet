---
title: "AdaBoostR2RegressionOptions"
description: "Configuration options for the AdaBoost R2 regression algorithm."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the AdaBoost R2 regression algorithm.

## For Beginners

AdaBoost (Adaptive Boosting) is like having a team of experts
(decision trees) working together to solve a problem. Each expert specializes in fixing
the mistakes made by previous experts. The "R2" indicates this is a version designed
specifically for regression problems (predicting continuous values like prices or temperatures)
rather than classification problems (categorizing data into groups).

## How It Works

AdaBoost R2 is an ensemble learning method that combines multiple decision trees
to create a more powerful regression model.

This class inherits from `DecisionTreeOptions`, which means it includes all the
configuration options for decision trees plus additional options specific to AdaBoost R2.

## Properties

| Property | Summary |
|:-----|:--------|
| `NumberOfEstimators` | Gets or sets the number of decision tree estimators (weak learners) to use in the ensemble. |

