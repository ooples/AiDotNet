---
title: "IEnsembleDifficultyEstimator<T, TInput, TOutput>"
description: "Interface for ensemble difficulty estimators that combine multiple estimators."
section: "API Reference"
---

`Interfaces` · `AiDotNet.CurriculumLearning.Interfaces`

Interface for ensemble difficulty estimators that combine multiple estimators.

## Properties

| Property | Summary |
|:-----|:--------|
| `Estimators` | Gets the individual estimators in this ensemble. |
| `Weights` | Gets or sets the weights for each estimator in the ensemble. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddEstimator(IDifficultyEstimator<,,>,)` | Adds an estimator to the ensemble. |
| `RemoveEstimator(Int32)` | Removes an estimator from the ensemble. |

