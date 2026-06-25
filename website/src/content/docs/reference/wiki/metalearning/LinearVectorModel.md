---
title: "LinearVectorModel"
description: "A simple linear model mapping Matrix input to Vector output, useful for meta-learning examples and testing."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Models`

A simple linear model mapping Matrix input to Vector output, useful for meta-learning examples and testing.

## For Beginners

This is a simple linear model used mainly for testing and
demonstrating meta-learning algorithms. It takes a matrix of inputs and produces predictions
using a straightforward linear formula (y = inputs * weights + bias). Its simplicity makes
it ideal for verifying that meta-learning algorithms like MAML work correctly before
scaling up to more complex models.

## How It Works

Computes y = X * w + b where w is a weight vector and b is a bias scalar.
Provides gradient computation via MSE loss for use with gradient-based meta-learners.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LinearVectorModel(Int32,Double)` | Creates a new linear model with the given input dimension. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DefaultLossFunction` |  |
| `ParameterCount` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGradients(Vector<Double>,Double)` |  |
| `ComputeGradients(Matrix<Double>,Vector<Double>,ILossFunction<Double>)` |  |
| `DeepCopy` |  |
| `Deserialize(Byte[])` |  |
| `GetActiveFeatureIndices` |  |
| `GetFeatureImportance` |  |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `IsFeatureUsed(Int32)` |  |
| `LoadModel(String)` |  |
| `LoadState(Stream)` |  |
| `Predict(Matrix<Double>)` |  |
| `SaveModel(String)` |  |
| `SaveState(Stream)` |  |
| `Serialize` |  |
| `SetActiveFeatureIndices(IEnumerable<Int32>)` |  |
| `SetParameters(Vector<Double>)` |  |
| `Train(Matrix<Double>,Vector<Double>)` |  |
| `WithParameters(Vector<Double>)` |  |

