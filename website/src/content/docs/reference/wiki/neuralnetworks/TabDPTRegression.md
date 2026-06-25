---
title: "TabDPTRegression<T>"
description: "TabDPT implementation for regression tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.Tabular`

TabDPT implementation for regression tasks.

## For Beginners

Use TabDPT for regression when:

- You need to predict continuous values from tabular data
- You want to leverage foundation model capabilities
- Your data has complex feature interactions

Example:

## How It Works

TabDPTRegression applies foundation model concepts to tabular regression,
leveraging pre-trained representations for continuous value prediction.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TabDPTRegression(Int32,Int32,TabDPTOptions<>)` | Initializes a new instance of the TabDPTRegression class. |

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

