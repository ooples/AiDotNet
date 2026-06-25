---
title: "MultiOutputClassifier<T>"
description: "Multi-output classifier for independent multi-label classification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Classification.Meta`

Multi-output classifier for independent multi-label classification.

## For Beginners

MultiOutputClassifier is the simplest multi-label approach:

For labels A, B, C:

- Classifier 1: Predict A using features X
- Classifier 2: Predict B using features X
- Classifier 3: Predict C using features X

Each classifier is completely independent.

When to use:

- When labels are truly independent
- As a simple baseline for multi-label problems
- When you don't need to model label correlations

Note: Unlike ClassifierChain, this does NOT capture label dependencies.

## How It Works

MultiOutputClassifier fits one classifier per target label, treating
each label as independent of the others.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MultiOutputClassifier` | Initializes a new instance with default settings using Gaussian Naive Bayes as the base estimator. |
| `MultiOutputClassifier(Func<IClassifier<>>,MetaClassifierOptions<>,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of the MultiOutputClassifier class. |

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
| `Predict(Matrix<>)` |  |
| `PredictLogProbabilities(Matrix<>)` |  |
| `PredictMultiLabel(Matrix<>)` | Predicts multi-label output for the given input. |
| `PredictMultiLabelProbabilities(Matrix<>)` |  |
| `PredictProbabilities(Matrix<>)` |  |
| `Serialize` |  |
| `Train(Matrix<>,Vector<>)` | Standard training method - converts single labels to multi-label format. |
| `TrainMultiLabel(Matrix<>,Matrix<>)` | Returns the model type identifier for this classifier. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_classifiers` | The classifiers, one per output. |

