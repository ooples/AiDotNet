---
title: "TabPFNRegression<T>"
description: "TabPFN implementation for regression tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.Tabular`

TabPFN implementation for regression tasks.

## For Beginners

TabPFN regression works similarly to classification:

1. First, call SetContext() with your training data and target values
2. Then, call Predict() with test data
3. The model uses attention to "learn" the regression pattern from context

Example:

## How It Works

TabPFNRegression uses in-context learning for tabular regression.
It takes training data as context and makes predictions on test data
in a single forward pass using attention mechanisms.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TabPFNRegression(Int32,Int32,TabPFNOptions<>)` | Initializes a new instance of the TabPFNRegression class. |

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
| `PredictEnsemble(Tensor<>,Int32,Matrix<Int32>)` | Predicts with ensemble averaging over multiple permutations. |
| `ResetState` | Resets internal state. |
| `SetContext(Tensor<>,Tensor<>)` | Sets the context (training) data for in-context learning. |
| `TrainStep(Tensor<>,Tensor<>,,Matrix<Int32>)` | Performs a single training step (for fine-tuning). |
| `UpdateParameters()` | Updates all parameters. |

