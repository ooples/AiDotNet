---
title: "OneVsOneClassifier<T>"
description: "One-vs-One classifier for multi-class classification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Classification.Meta`

One-vs-One classifier for multi-class classification.

## For Beginners

One-vs-One trains a classifier for every pair of classes:

For 3 classes (A, B, C):

- Classifier 1: A vs B
- Classifier 2: A vs C
- Classifier 3: B vs C

For K classes, this requires K*(K-1)/2 classifiers.

For prediction, each classifier votes for one class, and
the class with the most votes wins.

Advantages:

- Each classifier is trained on balanced binary problems
- Works well with SVM and other pairwise classifiers
- Good for small to medium number of classes

Disadvantages:

- Requires many classifiers for large K (K*(K-1)/2)
- Slower training for many classes

## How It Works

Trains one binary classifier for each pair of classes.
Uses voting to determine the final prediction.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OneVsOneClassifier` | Initializes a new instance with default settings using Gaussian Naive Bayes as the base estimator. |
| `OneVsOneClassifier(Func<IClassifier<>>,MetaClassifierOptions<>,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of the OneVsOneClassifier class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `CreateNewInstance` |  |
| `Deserialize(Byte[])` |  |
| `ExtractPairData(Matrix<>,Vector<>,Int32,Int32)` | Extracts samples belonging to a pair of classes. |
| `Predict(Matrix<>)` |  |
| `PredictLogProbabilities(Matrix<>)` |  |
| `PredictProbabilities(Matrix<>)` |  |
| `Serialize` |  |
| `Train(Matrix<>,Vector<>)` | Returns the model type identifier for this classifier. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_classPairs` | The class pair indices for each estimator. |
| `_estimators` | The binary classifiers for each pair of classes. |

