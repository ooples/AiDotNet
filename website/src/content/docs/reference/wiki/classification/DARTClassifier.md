---
title: "DARTClassifier<T>"
description: "DART (Dropouts meet Multiple Additive Regression Trees) classifier."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Classification.Boosting`

DART (Dropouts meet Multiple Additive Regression Trees) classifier.

## For Beginners

DART is like gradient boosting classifier with a twist - it randomly
"forgets" some of its trees when learning new ones. This prevents the model from becoming
too specialized and helps it work better on new data.

Key concepts:

- Dropout: Randomly removing trees during training (like dropout in neural networks)
- Normalization: Adjusting predictions after dropout to maintain correct scale
- Ensemble: The final prediction uses all trees (no dropout at prediction time)

When to use DART over regular gradient boosting:

- Your model overfits (training error low, validation error high)
- You want more robust predictions
- You have enough time (DART is slower than regular boosting)

## How It Works

DART applies dropout regularization to gradient boosting for classification. During each
iteration, a random subset of existing trees is dropped, and the new tree is fitted to
residuals computed only from the non-dropped trees. This prevents overfitting.

Reference: Rashmi, K.V. & Gilad-Bachrach, R. (2015). "DART: Dropouts meet Multiple
Additive Regression Trees". AISTATS 2015.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DARTClassifier(DARTClassifierOptions<>,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of DART classifier. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumberOfTrees` | Gets the number of trees in the ensemble. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateFeatureImportances(Int32)` | Calculates feature importances from all trees. |
| `ComputeLoss(Vector<>,Vector<>)` | Computes log loss for current predictions. |
| `CountBinomialDrops(Int32,Double)` | Counts drops using binomial distribution. |
| `CreateNewInstance` |  |
| `Deserialize(Byte[])` |  |
| `GetModelMetadata` |  |
| `Predict(Matrix<>)` | Predicts class labels for input samples. |
| `PredictProbabilities(Matrix<>)` | Predicts class probabilities for input samples. |
| `SelectNumDropout(Int32)` | Selects number of trees to drop based on dropout type. |
| `Serialize` |  |
| `Sigmoid()` | Sigmoid function for converting log-odds to probability. |
| `Train(Matrix<>,Vector<>)` | Returns the model type identifier for this classifier. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_initPrediction` | Initial log-odds prediction. |
| `_options` | Configuration options. |
| `_random` | Random number generator. |
| `_treeWeights` | Tree weights (may differ after dropout normalization). |
| `_trees` | Individual regression trees (trained on pseudo-residuals). |

