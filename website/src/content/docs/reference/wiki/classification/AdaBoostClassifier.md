---
title: "AdaBoostClassifier<T>"
description: "AdaBoost (Adaptive Boosting) classifier that combines weak learners."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Classification.Ensemble`

AdaBoost (Adaptive Boosting) classifier that combines weak learners.

## For Beginners

AdaBoost works like a learning system that focuses on its mistakes:

1. Train a simple classifier on the data
2. See which samples were misclassified
3. Give those samples higher importance
4. Train another classifier with the new importance weights
5. Repeat many times
6. Combine all classifiers with voting

This creates a powerful classifier from many weak ones.

## How It Works

AdaBoost iteratively trains weak classifiers on re-weighted versions of the data,
where incorrectly classified samples receive higher weights in subsequent iterations.
The final prediction is a weighted vote of all weak learners.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AdaBoostClassifier(AdaBoostClassifierOptions<>,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of the AdaBoostClassifier class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Options` | Gets the AdaBoost specific options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `CreateNewInstance` |  |
| `Deserialize(Byte[])` |  |
| `GetModelMetadata` |  |
| `Predict(Matrix<>)` |  |
| `PredictProbabilities(Matrix<>)` |  |
| `SampleWithWeights(Matrix<>,Vector<>,Vector<>)` | Samples data with replacement based on weights. |
| `Serialize` |  |
| `Train(Matrix<>,Vector<>)` | Returns the model type identifier for this classifier. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_estimatorWeights` | Weights for each estimator (based on their accuracy). |
| `_random` | Random number generator. |

