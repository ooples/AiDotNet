---
title: "BaggingClassifier<T>"
description: "Bagging (Bootstrap Aggregating) classifier."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Classification.Meta`

Bagging (Bootstrap Aggregating) classifier.

## For Beginners

Bagging is a technique to reduce overfitting:

1. Create N bootstrap samples (random samples with replacement)
2. Train one classifier on each sample
3. For prediction, each classifier votes
4. Final prediction is the majority vote

Benefits:

- Reduces variance (less overfitting)
- Works well with high-variance classifiers like decision trees
- Easily parallelizable

When to use:

- When your base classifier tends to overfit
- When you want more robust predictions
- As a simpler alternative to boosting

## How It Works

Bagging trains multiple classifiers on bootstrap samples of the training data
and combines their predictions through voting.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BaggingClassifier` | Initializes a new instance with default settings using Gaussian Naive Bayes as the base estimator. |
| `BaggingClassifier(Func<IClassifier<>>,BaggingClassifierOptions<>,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of the BaggingClassifier class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Options` | Gets the bagging-specific options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `CreateBootstrapSample(Matrix<>,Vector<>,Int32)` | Creates a bootstrap sample from the data. |
| `CreateNewInstance` |  |
| `Deserialize(Byte[])` |  |
| `GetModelMetadata` |  |
| `Predict(Matrix<>)` |  |
| `PredictLogProbabilities(Matrix<>)` |  |
| `PredictProbabilities(Matrix<>)` |  |
| `SampleFeaturesRandomly(Matrix<>,Int32)` | Randomly samples a subset of features. |
| `Serialize` |  |
| `Train(Matrix<>,Vector<>)` | Returns the model type identifier for this classifier. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_estimators` | The ensemble of classifiers. |
| `_featureIndicesPerEstimator` | Feature indices selected for each estimator. |
| `_random` | Random number generator. |

