---
title: "MambularRegression<T>"
description: "Mambular implementation for regression tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.Tabular`

Mambular implementation for regression tasks.

## For Beginners

Use Mambular for regression when:

- You have many features and need efficient processing
- You want an alternative to transformer-based models
- Feature order/sequence matters for your task

Example:

## How It Works

MambularRegression applies State Space Models (Mamba architecture) to
tabular data for predicting continuous values.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MambularRegression(Int32,Int32,MambularOptions<>)` | Initializes a new instance of the MambularRegression class. |

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

