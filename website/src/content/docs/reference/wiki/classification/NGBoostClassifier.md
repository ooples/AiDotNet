---
title: "NGBoostClassifier<T>"
description: "NGBoost (Natural Gradient Boosting) classifier for probabilistic classification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Classification.Boosting`

NGBoost (Natural Gradient Boosting) classifier for probabilistic classification.

## For Beginners

Traditional classifiers give you a class prediction like
"this email is spam" with maybe a confidence score. But NGBoost gives you
properly calibrated probabilities - if it says "70% spam", then about 70% of
similar predictions will actually be spam.

Key benefits:

- Well-calibrated probability estimates
- Quantifies prediction uncertainty
- Uses natural gradients for stable, efficient learning
- Works well for both binary and multi-class problems

## How It Works

NGBoost is an algorithm for probabilistic prediction that uses natural gradients
to efficiently and directly optimize a proper scoring rule. For classification,
it predicts class probabilities that are well-calibrated.

Reference: Duan, T., et al. "NGBoost: Natural Gradient Boosting for Probabilistic
Prediction" (2019). https://arxiv.org/abs/1910.03225

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NGBoostClassifier(NGBoostClassifierOptions<>,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of NGBoostClassifier. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumberOfTrees` | Gets the number of trees in the ensemble. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateFeatureImportances(Int32)` | Calculates feature importances from all trees. |
| `ComputeCrossEntropyLoss(Vector<>[],Int32[])` | Computes cross-entropy loss for current predictions. |
| `ComputeNaturalGradients(Vector<>[],Vector<>[],Int32)` | Computes natural gradients using Fisher Information approximation. |
| `ComputeProbabilities(Vector<>[],Int32[])` | Computes probabilities from log-odds using softmax. |
| `CreateNewInstance` |  |
| `Deserialize(Byte[])` |  |
| `GetModelMetadata` |  |
| `GetSampleIndices(Int32)` | Gets sample indices for subsampling. |
| `Predict(Matrix<>)` | Predicts class labels for the input samples. |
| `PredictLogProbabilities(Matrix<>)` | Predicts log probabilities for the input samples. |
| `PredictProbabilities(Matrix<>)` | Predicts class probabilities for the input samples. |
| `Serialize` |  |
| `Train(Matrix<>,Vector<>)` | Returns the model type identifier for this classifier. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_initialLogOdds` | Initial log-odds values for each class. |
| `_numClasses` | Number of classes. |
| `_options` | Configuration options. |
| `_random` | Random number generator. |
| `_trees` | Base learners for each class's log-odds. |

