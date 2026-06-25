---
title: "LinearSupportVectorClassifier<T>"
description: "Linear Support Vector Classifier optimized for linear classification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Classification.SVM`

Linear Support Vector Classifier optimized for linear classification.

## For Beginners

Linear SVC is a simplified version of SVM that only draws straight lines to separate classes.
It's much faster to train than the regular SVC because it doesn't need to compute kernel
values between all pairs of training points.

Use Linear SVC when:

- You have a large dataset (thousands of samples)
- Your data is linearly separable or nearly so
- You need fast training and prediction
- You have high-dimensional data (many features)

Example use cases:

- Text classification (spam detection, sentiment)
- Document categorization
- High-dimensional bioinformatics data

## How It Works

This implementation uses a primal formulation with stochastic gradient descent (SGD)
for efficient training on large datasets. Unlike the standard SVC which uses the kernel
trick, this classifier works directly in the original feature space.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LinearSupportVectorClassifier(SVMOptions<>,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of the LinearSupportVectorClassifier class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `ComputeLinearOutput(Vector<>)` | Computes the linear output w · x + b. |
| `CreateNewInstance` |  |
| `DecisionFunction(Matrix<>)` |  |
| `Deserialize(Byte[])` |  |
| `GetModelMetadata` |  |
| `Predict(Matrix<>)` |  |
| `PredictProbabilities(Matrix<>)` |  |
| `Serialize` |  |
| `ShuffleArray(Int32[])` | Shuffles an array in place. |
| `Train(Matrix<>,Vector<>)` | Returns the model type identifier for this classifier. |
| `TrainSGD(Matrix<>,Vector<>)` | Trains using Stochastic Gradient Descent with hinge loss. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_bias` | Bias term (intercept) for the linear classifier. |
| `_random` | Random number generator for SGD. |
| `_weights` | Weight vector for linear classification. |

