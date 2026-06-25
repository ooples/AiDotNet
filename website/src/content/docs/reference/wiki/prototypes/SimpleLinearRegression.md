---
title: "SimpleLinearRegression<T>"
description: "Simple linear regression model for prototype validation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Prototypes`

Simple linear regression model for prototype validation.
Demonstrates GPU acceleration for traditional machine learning algorithms.

## How It Works

Linear regression model: y = X @ weights + bias

This implementation uses vectorized operations throughout, enabling GPU
acceleration when using float type with GPU engine enabled.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SimpleLinearRegression(Int32)` | Initializes a new instance of the SimpleLinearRegression. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsTrained` | Gets whether the model has been trained. |
| `NumFeatures` | Gets the number of features. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeMSE(PrototypeVector<>,PrototypeVector<>)` | Computes Mean Squared Error. |
| `ComputeR2Score(PrototypeVector<>,PrototypeVector<>)` | Computes R² score (coefficient of determination). |
| `GetBias` | Gets the learned bias. |
| `GetWeights` | Gets the learned weights. |
| `MatrixVectorMultiply(PrototypeVector<>,PrototypeVector<>,Int32,Int32)` | Matrix-vector multiplication: result = matrix @ vector Matrix is stored in row-major order (flattened). |
| `MatrixVectorMultiplyTranspose(PrototypeVector<>,PrototypeVector<>,Int32,Int32)` | Matrix-vector multiplication with transposed matrix: result = matrix^T @ vector |
| `Predict(PrototypeVector<>)` | Makes a prediction for a single sample. |
| `PredictBatch(PrototypeVector<>,Int32)` | Makes predictions for a batch of samples. |
| `ToString` | Returns a string representation of the model. |
| `Train(PrototypeVector<>,PrototypeVector<>,Int32,Double,Int32,Boolean)` | Trains the model using gradient descent. |

