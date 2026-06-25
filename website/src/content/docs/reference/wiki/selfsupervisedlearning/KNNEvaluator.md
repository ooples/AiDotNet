---
title: "KNNEvaluator<T>"
description: "k-Nearest Neighbors (k-NN) evaluation for SSL representation quality."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SelfSupervisedLearning.Evaluation`

k-Nearest Neighbors (k-NN) evaluation for SSL representation quality.

## For Beginners

k-NN evaluation is a simple way to test SSL representations
without any training. We classify each test sample based on the labels of its k nearest
neighbors in the representation space.

## How It Works

**Why k-NN evaluation?**

**Typical protocol:**

**Common settings:** k=20, cosine similarity or L2 distance

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `KNNEvaluator(INeuralNetwork<>,Int32,Boolean,Double)` | Initializes a new instance of the KNNEvaluator class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DefaultLossFunction` |  |
| `K` | Gets the k value for k-NN. |

## Methods

| Method | Summary |
|:-----|:--------|
| `DeepCopy` |  |
| `Evaluate(Tensor<>,Int32[])` | Evaluates k-NN classification on test data. |
| `EvaluateMultipleK(Tensor<>,Int32[],Int32[])` | Evaluates k-NN with different k values. |
| `Fit(Tensor<>,Int32[])` | Fits the k-NN classifier with training data. |
| `GetParameters` |  |
| `Predict(Tensor<>)` |  |
| `PredictClasses(Tensor<>)` | Predicts classes for multiple test samples. |
| `SetParameters(Vector<>)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `WithParameters(Vector<>)` |  |

