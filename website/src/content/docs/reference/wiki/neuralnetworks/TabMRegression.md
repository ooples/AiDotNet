---
title: "TabMRegression<T>"
description: "TabM implementation for regression tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.Tabular`

TabM implementation for regression tasks.

## For Beginners

Use this model when you want to predict continuous values
(like prices, temperatures, etc.) with the benefits of an ensemble.

How predictions work:

1. Input passes through shared layers with per-member modulation
2. Each ensemble member produces a prediction
3. Predictions are averaged for the final output
4. Variance across members provides uncertainty estimate

Benefits over single models:

- Better generalization through ensemble averaging
- Built-in uncertainty quantification
- More robust to outliers
- Comparable speed to single models

Example:

## How It Works

TabMRegression uses the TabM architecture with BatchEnsemble layers for regression.
It averages predictions across ensemble members and can provide uncertainty estimates.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TabMRegression(Int32,Int32,TabMOptions<>)` | Initializes a new instance of the TabMRegression class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `OutputDimension` | Gets the output dimension. |
| `ParameterCount` | Gets the total number of trainable parameters. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeCalibrationMetrics(Tensor<>,Tensor<>)` | Computes calibration metrics for uncertainty estimates. |
| `ComputeMAELoss(Tensor<>,Tensor<>)` | Computes the Mean Absolute Error (MAE) loss. |
| `ComputeMSELoss(Tensor<>,Tensor<>)` | Computes the Mean Squared Error (MSE) loss. |
| `ComputeNLL(Tensor<>,Tensor<>)` | Computes Negative Log Likelihood assuming Gaussian predictions. |
| `ComputeR2Score(Tensor<>,Tensor<>)` | Computes the R² score (coefficient of determination). |
| `ComputeRMSE(Tensor<>,Tensor<>)` | Computes the Root Mean Squared Error (RMSE). |
| `Forward(Tensor<>)` | Performs the forward pass to get predictions (per member). |
| `GetParameters` | Gets all parameters including the regression head. |
| `Predict(Tensor<>)` | Predicts values (averaged across ensemble members). |
| `PredictWithUncertainty(Tensor<>)` | Predicts values with uncertainty estimates. |
| `ResetState` | Resets internal state. |
| `TrainStep(Tensor<>,Tensor<>,)` | Performs a single training step using MSE loss. |
| `UpdateParameters()` | Updates all parameters including the regression head. |

