---
title: "SAINTRegression<T>"
description: "SAINT implementation for regression tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.Tabular`

SAINT implementation for regression tasks.

## For Beginners

Use SAINT for regression when:

- You have tabular data with mixed numerical and categorical features
- You believe similar samples in your data share useful patterns
- You want state-of-the-art performance on tabular regression

Example:

## How It Works

SAINTRegression applies both column attention (over features) and row attention
(over samples in a batch) for predicting continuous values on tabular data.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SAINTRegression(Int32,Int32,SAINTOptions<>)` | Initializes a new instance of the SAINTRegression class. |

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
| `ComputeR2Score(Tensor<>,Tensor<>,Matrix<Int32>)` | Computes the R² score (coefficient of determination). |
| `ComputeRMSE(Tensor<>,Tensor<>,Matrix<Int32>)` | Computes the Root Mean Squared Error. |
| `Forward(Tensor<>,Matrix<Int32>)` | Performs the forward pass to get predictions. |
| `Predict(Tensor<>,Matrix<Int32>)` | Makes predictions (alias for Forward). |
| `ResetState` | Resets internal state. |
| `TrainStep(Tensor<>,Tensor<>,,Matrix<Int32>)` | Performs a single training step. |
| `UpdateParameters()` | Updates all parameters. |

