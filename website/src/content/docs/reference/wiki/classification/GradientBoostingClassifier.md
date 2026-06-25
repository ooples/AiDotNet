---
title: "GradientBoostingClassifier<T>"
description: "Gradient Boosting classifier that builds trees sequentially to correct errors."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Classification.Ensemble`

Gradient Boosting classifier that builds trees sequentially to correct errors.

## For Beginners

Gradient Boosting is one of the most powerful machine learning algorithms:

How it works:

1. Start with an initial prediction
2. Calculate how wrong we are
3. Train a tree to predict our mistakes
4. Add a fraction of this tree's predictions
5. Repeat, each time correcting remaining errors

Key insight: Each tree fixes what previous trees got wrong!

Tips for best results:

- Use lower learning_rate with more n_estimators
- Keep max_depth small (3-5) unlike Random Forest
- Consider subsample less than 1.0 for regularization

## How It Works

Gradient Boosting builds an additive model in a forward stage-wise fashion.
At each stage, a regression tree is fit on the negative gradient of the loss function.
For classification, this uses log loss (deviance) or exponential loss.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GradientBoostingClassifier(GradientBoostingClassifierOptions<>,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of the GradientBoostingClassifier class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `LeafCount` |  |
| `MaxDepth` |  |
| `NodeCount` |  |
| `Options` | Gets the Gradient Boosting specific options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateTotalLeafCount` | Calculates the total number of leaf nodes. |
| `CalculateTotalNodeCount` | Calculates the total number of nodes. |
| `Clone` |  |
| `CreateNewInstance` |  |
| `Deserialize(Byte[])` |  |
| `GetModelMetadata` |  |
| `Predict(Matrix<>)` |  |
| `PredictProbabilities(Matrix<>)` |  |
| `Serialize` |  |
| `Sigmoid()` | Computes the sigmoid function. |
| `SubsampleData(Matrix<>,Vector<>)` | Subsamples data for stochastic gradient boosting. |
| `Train(Matrix<>,Vector<>)` | Returns the model type identifier for this classifier. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_initPrediction` | Initial prediction (prior). |
| `_leafResidualMeans` | Mean residual values for each tree's leaf predictions. |
| `_random` | Random number generator. |

