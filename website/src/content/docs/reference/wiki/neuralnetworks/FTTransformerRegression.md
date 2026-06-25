---
title: "FTTransformerRegression<T>"
description: "FT-Transformer implementation for regression tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.Tabular`

FT-Transformer implementation for regression tasks.

## For Beginners

Use this model when you want to predict continuous numbers
(like house prices, temperatures, stock returns, etc.).

How it works:

1. Features are tokenized and processed by transformer layers
2. The [CLS] token captures information from all features
3. A linear layer maps the [CLS] representation to the output value(s)

Example:

## How It Works

FTTransformerRegression applies the FT-Transformer architecture to regression problems.
It uses the [CLS] token output with a linear regression head to predict continuous values.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FTTransformerRegression(Int32,Int32,FTTransformerOptions<>)` | Initializes a new instance of the FTTransformerRegression class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `OutputDimension` | Gets the output dimension (number of target values to predict). |
| `ParameterCount` | Gets the total number of trainable parameters including the regression head. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeMAELoss(Tensor<>,Tensor<>)` | Computes the Mean Absolute Error (MAE) loss. |
| `ComputeMAPE(Tensor<>,Tensor<>,Matrix<Int32>)` | Computes the Mean Absolute Percentage Error (MAPE). |
| `ComputeMSELoss(Tensor<>,Tensor<>)` | Computes the Mean Squared Error (MSE) loss. |
| `ComputeR2Score(Tensor<>,Tensor<>,Matrix<Int32>)` | Computes the R² score (coefficient of determination). |
| `ComputeRMSE(Tensor<>,Tensor<>,Matrix<Int32>)` | Computes the Root Mean Squared Error (RMSE). |
| `Forward(Tensor<>)` | Performs the forward pass with numerical features only. |
| `Forward(Tensor<>,Matrix<Int32>)` | Performs the forward pass to get predictions. |
| `GetParameters` | Gets all parameters including the regression head. |
| `Predict(Tensor<>)` | Alias for Forward with numerical features only. |
| `Predict(Tensor<>,Matrix<Int32>)` | Alias for Forward - makes predictions on the input data. |
| `ResetState` | Resets internal state. |
| `SetParameters(Vector<>)` | Sets all parameters including the regression head. |
| `TrainStep(Tensor<>,Tensor<>,,Matrix<Int32>)` | Performs a single training step using MSE loss. |
| `UpdateParameters()` | Updates all parameters including the regression head. |

