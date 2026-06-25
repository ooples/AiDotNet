---
title: "LinearClassifierBase<T>"
description: "Provides a base implementation for linear classifiers."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Classification.Linear`

Provides a base implementation for linear classifiers.

## For Beginners

Linear classifiers are one of the simplest forms of machine learning:

How they work:

1. Each feature gets a weight (importance score)
2. Multiply each feature by its weight and sum them up
3. Add a bias term
4. If the result is positive, predict one class; otherwise, the other

The training process adjusts the weights to correctly classify
training examples.

Advantages:

- Fast to train and predict
- Work well with many features
- Easy to interpret (weight = feature importance)
- Often surprisingly effective

## How It Works

Linear classifiers learn a linear decision function: f(x) = w * x + b
where w is the weight vector and b is the bias (intercept).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LinearClassifierBase(LinearClassifierOptions<>,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of the LinearClassifierBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Intercept` | The learned intercept (bias) term. |
| `Options` | Gets the linear classifier specific options. |
| `Random` | Random number generator for shuffling. |
| `Weights` | The learned weight vector. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGradients(Vector<>,)` |  |
| `ApplyL1Gradient(,)` | Applies L1 regularization gradient to the weights. |
| `ApplyL2Gradient(,)` | Applies L2 regularization gradient to the weights. |
| `ComputeGradients(Matrix<>,Vector<>,ILossFunction<>)` |  |
| `DecisionFunction(Vector<>)` | Computes the decision function value for a single sample. |
| `DecisionFunctionBatch(Matrix<>)` | Computes decision function values for all samples. |
| `Deserialize(Byte[])` |  |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `InitializeWeights` | Initializes the weights before training. |
| `Predict(Matrix<>)` |  |
| `PredictLogProbabilities(Matrix<>)` |  |
| `PredictProbabilities(Matrix<>)` |  |
| `Serialize` |  |
| `SetParameters(Vector<>)` |  |
| `ShuffleIndices(Int32)` | Shuffles the training data indices. |
| `Sigmoid()` | Computes the sigmoid function. |
| `WithParameters(Vector<>)` |  |

