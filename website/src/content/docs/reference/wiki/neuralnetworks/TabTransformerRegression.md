---
title: "TabTransformerRegression<T>"
description: "TabTransformer implementation for regression tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.Tabular`

TabTransformer implementation for regression tasks.

## For Beginners

Use TabTransformer for regression when:

- You have important categorical features
- You believe there are interactions between categorical features
- You want to predict continuous values (prices, amounts, etc.)

Example:

## How It Works

TabTransformerRegression applies transformer attention to categorical features
for predicting continuous values. Numerical features are concatenated after
the categorical embeddings are transformed.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TabTransformerRegression(Int32,Int32,TabTransformerOptions<>)` | Initializes a new instance of the TabTransformerRegression class. |

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
| `ComputeR2Score(Tensor<>,Tensor<>,Matrix<Int32>)` | Computes the R² score. |
| `ComputeRMSE(Tensor<>,Tensor<>,Matrix<Int32>)` | Computes the Root Mean Squared Error. |
| `Forward(Tensor<>,Matrix<Int32>)` | Performs the forward pass to get predictions. |
| `Predict(Tensor<>,Matrix<Int32>)` | Makes predictions (alias for Forward). |
| `ResetState` | Resets internal state. |
| `TrainStep(Tensor<>,Tensor<>,,Matrix<Int32>)` | Performs a single training step. |
| `UpdateParameters()` | Updates all parameters. |

