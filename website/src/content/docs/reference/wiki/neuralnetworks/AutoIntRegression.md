---
title: "AutoIntRegression<T>"
description: "AutoInt implementation for regression tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.Tabular`

AutoInt implementation for regression tasks.

## For Beginners

Use AutoInt for regression when:

- Feature interactions matter for prediction
- You don't want to manually engineer feature crosses
- You want interpretable interaction patterns

Example:

## How It Works

AutoIntRegression uses multi-head self-attention to automatically learn
feature interactions for regression on tabular data.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AutoIntRegression(Int32,Int32,AutoIntOptions<>)` | Initializes a new instance of the AutoIntRegression class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `OutputDimension` | Gets the output dimension. |
| `ParameterCount` | Gets the total number of trainable parameters. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeMAELoss(Tensor<>,Tensor<>)` | Computes the Mean Absolute Error loss. |
| `ComputeMSELoss(Tensor<>,Tensor<>)` | Computes the Mean Squared Error loss. |
| `ComputeR2Score(Tensor<>,Tensor<>)` | Computes the R² score (coefficient of determination) from precomputed predictions. |
| `ComputeR2Score(Tensor<>,Tensor<>,Matrix<Int32>)` | Computes the R² score by running a forward pass on the given features. |
| `ComputeRMSE(Tensor<>,Tensor<>)` | Computes the Root Mean Squared Error from precomputed predictions. |
| `ComputeRMSE(Tensor<>,Tensor<>,Matrix<Int32>)` | Computes the Root Mean Squared Error by running a forward pass on the given features. |
| `Forward(Tensor<>,Matrix<Int32>)` | Performs the forward pass to get predictions. |
| `Predict(Tensor<>,Matrix<Int32>)` | Makes predictions (alias for Forward). |
| `ResetState` | Resets internal state. |
| `TrainStep(Tensor<>,Tensor<>,,Matrix<Int32>)` | Performs a single training step. |
| `UpdateParameters()` | Updates all parameters. |

