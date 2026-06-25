---
title: "OrdinalClassifierBase<T>"
description: "Base class for ordinal classification models."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Classification.Ordinal`

Base class for ordinal classification models.

## For Beginners

Ordinal classification (also called ordinal regression) is used when
your target variable has a natural order. Examples include:

- Star ratings (1, 2, 3, 4, 5 stars)
- Education levels (High School, Bachelor's, Master's, PhD)
- Disease severity (Mild, Moderate, Severe)
- Likert scale responses (Strongly Disagree to Strongly Agree)

## How It Works

Unlike regular classification where all misclassifications are equally bad,
in ordinal classification being "close" to the true class is better than being far away.
Predicting 4 stars when the truth is 5 stars is a smaller error than predicting 1 star.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OrdinalClassifierBase(ClassifierOptions<>,IRegularization<,Matrix<>,Vector<>>,ILossFunction<>)` | Initializes a new instance of OrdinalClassifierBase. |

## Properties

| Property | Summary |
|:-----|:--------|
| `OrderedClasses` | Gets the ordered class labels. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ExtractOrderedClasses(Vector<>)` | Extracts ordered class labels from the training labels. |
| `GetClassIndex()` | Converts a class label to its ordinal index. |
| `InferTaskType(Vector<>)` | Infers the task type for ordinal classification. |
| `PredictCumulativeProbabilities(Matrix<>)` | Predicts cumulative probabilities P(Y ≤ k) for each class threshold. |
| `PredictProbabilities(Matrix<>)` | Predicts class probabilities for each ordinal class. |
| `Sigmoid()` | Computes the sigmoid function. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_thresholds` | The learned thresholds that separate ordinal classes. |

