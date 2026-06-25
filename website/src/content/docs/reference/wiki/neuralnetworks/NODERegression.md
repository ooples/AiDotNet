---
title: "NODERegression<T>"
description: "NODE implementation for regression tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.Tabular`

NODE implementation for regression tasks.

## For Beginners

Use NODE for regression when you want:

- Interpretable models (see feature importance via GetFeatureImportance())
- Tree-based structure with neural network trainability
- Good performance on tabular data with continuous targets

Example:

## How It Works

NODERegression uses an ensemble of differentiable oblivious decision trees
for predicting continuous values on tabular data.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NODERegression(Int32,Int32,NODEOptions<>)` | Initializes a new instance of the NODERegression class. |

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

