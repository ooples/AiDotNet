---
title: "GANDALFRegression<T>"
description: "GANDALF implementation for regression tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.Tabular`

GANDALF implementation for regression tasks.

## For Beginners

Use GANDALF for regression when you want:

- Automatic feature importance learning
- Interpretable predictions
- Good performance on tabular data with continuous targets

Example:

## How It Works

GANDALFRegression uses gated feature selection with neural decision trees
for predicting continuous values. The additive ensemble of trees directly
produces the regression output.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GANDALFRegression` | Initializes a new instance of the GANDALFRegression class with default configuration. |
| `GANDALFRegression(Int32,Int32,GANDALFOptions<>)` | Initializes a new instance of the GANDALFRegression class. |

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
| `ComputeR2Score(Tensor<>,Tensor<>)` | Computes the R² score (coefficient of determination). |
| `ComputeRMSE(Tensor<>,Tensor<>)` | Computes the Root Mean Squared Error. |
| `Forward(Tensor<>)` | Performs the forward pass to get predictions. |
| `Predict(Tensor<>)` | Makes predictions (alias for Forward). |
| `ResetState` | Resets internal state. |
| `TrainStep(Tensor<>,Tensor<>,)` | Performs a single training step. |
| `UpdateParameters()` | Updates all parameters. |

