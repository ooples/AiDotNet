---
title: "ClassifierChain<T>"
description: "Classifier Chain for multi-label classification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Classification.Meta`

Classifier Chain for multi-label classification.

## For Beginners

Classifier Chain captures label dependencies:

For labels A, B, C:

- Classifier 1: Predict A using features X
- Classifier 2: Predict B using features X + prediction of A
- Classifier 3: Predict C using features X + predictions of A and B

Benefits:

- Captures dependencies between labels
- Better than independent binary classifiers

Trade-offs:

- Order of chain matters (can use random order or learned order)
- Error propagation (early mistakes affect later predictions)

## How It Works

Classifier Chain transforms a multi-label problem into a chain of binary
classification problems, where each classifier uses the predictions of
previous classifiers as additional features.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ClassifierChain` | Initializes a new instance with default settings using Gaussian Naive Bayes as the base estimator. |
| `ClassifierChain(Func<IClassifier<>>,ClassifierChainOptions<>,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of the ClassifierChain class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `LabelNames` | Gets or sets the label names if available. |
| `NumLabels` | Gets the number of labels (same as number of classes for multi-label). |
| `Options` | Gets the chain-specific options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `CreateAugmentedFeatures(Matrix<>,Matrix<>,Int32)` | Creates augmented features by adding previous predictions. |
| `CreateNewInstance` |  |
| `Deserialize(Byte[])` |  |
| `DetermineOrder` | Determines the order of labels in the chain. |
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
| `_classifiers` | The classifiers in the chain, one per label. |
| `_order` | The order of labels in the chain. |

