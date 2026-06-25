---
title: "EnsembleClassifierBase<T>"
description: "Base class for ensemble classification methods that combine multiple classifiers."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Classification.Ensemble`

Base class for ensemble classification methods that combine multiple classifiers.

## For Beginners

Ensemble learning is like getting opinions from a group of experts instead of just one.

Imagine you want to predict if a movie will be successful. You could:

1. Ask just one expert (single classifier)
2. Ask many experts and combine their opinions (ensemble)

The second approach is usually more reliable because:

- Individual experts may have blind spots that others don't
- Combining diverse opinions often leads to better decisions
- Errors from one expert may be corrected by others

Common ensemble strategies:

- Bagging: Train on different random subsets of data
- Boosting: Train sequentially, focusing on mistakes
- Voting: Let classifiers vote on the answer

## How It Works

Ensemble methods combine multiple individual classifiers (base estimators) to produce
a more robust and accurate prediction than any single classifier could achieve.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EnsembleClassifierBase(ClassifierOptions<>,IRegularization<,Matrix<>,Vector<>>,ILossFunction<>)` | Initializes a new instance of the EnsembleClassifierBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Estimators` | The base estimators in the ensemble. |
| `FeatureImportances` | Gets or sets the feature importances aggregated across all estimators. |
| `NEstimators` | The number of estimators in the ensemble. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AggregateFeatureImportances` | Aggregates feature importances from all tree-based estimators. |
| `GetModelMetadata` |  |
| `PredictProbabilities(Matrix<>)` | Aggregates predictions from all estimators in the ensemble. |

