---
title: "KNeighborsClassifier<T>"
description: "K-Nearest Neighbors classifier that predicts based on the majority class of nearest neighbors."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Classification.Neighbors`

K-Nearest Neighbors classifier that predicts based on the majority class of nearest neighbors.

## For Beginners

KNN is one of the simplest machine learning algorithms. Think of it as "you are the
company you keep." To classify something new:

1. Find the k most similar training examples
2. Look at their classes
3. Return the most common class

Example: Classifying a fruit by its weight and color:

- New fruit: 150g, red
- 3 nearest neighbors: Apple (160g, red), Apple (145g, red), Cherry (10g, red)
- Wait! Cherry is much smaller, so it's not really "near"
- Actual 3 nearest: Apple, Apple, Orange -> Predicted: Apple

This is why feature scaling is important for KNN!

## How It Works

K-Nearest Neighbors (KNN) is a non-parametric, instance-based learning algorithm.
It stores all training data and classifies new samples by finding the k closest
training samples and voting on their class labels.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `KNeighborsClassifier(KNeighborsOptions<>,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of the KNeighborsClassifier class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Options` | Gets the KNN specific options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGradients(Vector<>,)` |  |
| `Clone` |  |
| `ComputeChebyshevDistance(Vector<>,Vector<>)` | Computes Chebyshev (L-infinity) distance. |
| `ComputeCosineDistance(Vector<>,Vector<>)` | Computes cosine distance (1 - cosine similarity). |
| `ComputeDistance(Vector<>,Vector<>)` | Computes the distance between two samples based on the configured metric. |
| `ComputeGradients(Matrix<>,Vector<>,ILossFunction<>)` |  |
| `ComputeManhattanDistance(Vector<>,Vector<>)` | Computes Manhattan (L1) distance. |
| `ComputeMinkowskiDistance(Vector<>,Vector<>,Double)` | Computes Minkowski distance with parameter p. |
| `CreateNewInstance` |  |
| `Deserialize(Byte[])` |  |
| `GetClassIndex()` | Gets the class index for a label. |
| `GetDistanceWeight()` | Computes weight based on distance (1/distance). |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `GetWeight()` | Gets the weight for a neighbor based on the weighting scheme. |
| `PredictProbabilities(Matrix<>)` |  |
| `PredictSampleProbabilities(Vector<>)` | Predicts class probabilities for a single sample. |
| `Serialize` |  |
| `SetParameters(Vector<>)` |  |
| `Train(Matrix<>,Vector<>)` | Returns the model type identifier for this classifier. |
| `WithParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_xTrain` | Stored training features. |
| `_yTrain` | Stored training labels. |

