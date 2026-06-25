---
title: "TabRRegression<T>"
description: "TabR implementation for regression tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.Tabular`

TabR implementation for regression tasks.

## For Beginners

TabR for regression is like estimating a home's price
by looking at similar homes that have sold recently.

Prediction process:

1. Encode the input features
2. Find similar training samples (neighbors)
3. Aggregate neighbor information using attention
4. Combine with encoded input to predict values

Benefits:

- Naturally handles local patterns (similar inputs → similar outputs)
- Can explain predictions by showing influential neighbors
- Works well with non-linear relationships

Example:

## How It Works

TabRRegression uses retrieval-augmented predictions for regression.
It finds similar training samples and uses their information to help
predict continuous values.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TabRRegression(Int32,Int32,TabROptions<>)` | Initializes a new instance of the TabRRegression class. |

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
| `Forward(Tensor<>,Vector<Int32>)` | Performs the forward pass to get predictions. |
| `GetPredictionExplanations(Tensor<>)` | Gets interpretability information: which neighbors influenced each prediction. |
| `Predict(Tensor<>)` | Makes predictions (alias for Forward). |
| `PredictWithConfidence(Tensor<>,Double)` | Predicts with confidence intervals based on neighbor variance. |
| `ResetState` | Resets internal state. |
| `TrainStep(Tensor<>,Tensor<>,,Vector<Int32>)` | Performs a single training step. |
| `UpdateParameters()` | Updates all parameters. |

