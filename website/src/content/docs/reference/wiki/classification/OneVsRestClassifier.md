---
title: "OneVsRestClassifier<T>"
description: "One-vs-Rest (also called One-vs-All) classifier for multi-class and multi-label classification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Classification.Meta`

One-vs-Rest (also called One-vs-All) classifier for multi-class and multi-label classification.

## For Beginners

One-vs-Rest is a simple strategy for multi-class and multi-label classification:

For 3 classes (A, B, C):

- Classifier 1: Is it A vs not-A?
- Classifier 2: Is it B vs not-B?
- Classifier 3: Is it C vs not-C?

For prediction, the class whose classifier gives the highest score wins.

Advantages:

- Simple and effective
- Trains K classifiers for K classes
- Easily parallelizable

Disadvantages:

- Class imbalance (one class vs all others)
- Classifiers don't see inter-class relationships

## How It Works

Trains one binary classifier per class, treating it as the positive class
and all other classes as the negative class. This approach works for both
multi-class (exactly one label per sample) and multi-label (zero or more labels per sample) problems.

**Multi-label usage:**
This classifier also supports multi-label classification via
`Matrix{`, along with the
`NumLabels` and `LabelNames` properties.
When using this class in a multi-label setting, prefer
`Matrix{` over
`Vector{`.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OneVsRestClassifier` | Initializes a new instance with default settings using Gaussian Naive Bayes as the base estimator. |
| `OneVsRestClassifier(Func<IClassifier<>>,MetaClassifierOptions<>,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of the OneVsRestClassifier class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `LabelNames` | Gets or sets the label names if available. |
| `NumLabels` | Gets the number of labels (same as number of classes for multi-label). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `CreateNewInstance` |  |
| `Deserialize(Byte[])` |  |
| `GetEstimatorScores(IClassifier<>,Matrix<>)` | Gets decision scores from an estimator. |
| `Predict(Matrix<>)` |  |
| `PredictLogProbabilities(Matrix<>)` |  |
| `PredictMultiLabel(Matrix<>)` |  |
| `PredictMultiLabelProbabilities(Matrix<>)` |  |
| `PredictProbabilities(Matrix<>)` |  |
| `Serialize` |  |
| `Train(Matrix<>,Vector<>)` | Trains the One-vs-Rest classifier on the provided data. |
| `TrainMultiLabel(Matrix<>,Matrix<>)` | Returns the model type identifier for this classifier. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_estimators` | The binary classifiers, one per class. |

