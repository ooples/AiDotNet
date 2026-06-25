---
title: "TabNetRegression<T>"
description: "TabNet implementation for regression tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.Tabular`

TabNet implementation for regression tasks.

## For Beginners

Use TabNetRegression when you want to predict a number
(continuous value) from tabular data.

Example use cases:

- Predicting house prices from features like square footage, bedrooms, location
- Forecasting sales based on historical data and market indicators
- Estimating customer lifetime value from demographic and behavioral data

Key features:

- **Automatic Feature Selection**: Learns which columns in your data matter most
- **Interpretability**: You can see exactly which features the model used
- **No Feature Engineering**: Often works well without manual feature preprocessing
- **Competitive Performance**: Matches or beats gradient boosting methods

Basic usage:

## How It Works

TabNetRegression extends the TabNet architecture for predicting continuous values.
It uses the same attention-based feature selection mechanism but with an output
layer suitable for regression.

Reference: "TabNet: Attentive Interpretable Tabular Learning" (Arik & Pfister, AAAI 2021)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TabNetRegression(Int32,Int32,TabNetOptions<>)` | Initializes a new instance of the TabNetRegression class for multi-output regression. |
| `TabNetRegression(Int32,TabNetOptions<>)` | Initializes a new instance of the TabNetRegression class for single-output regression. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeMAELoss(Tensor<>,Tensor<>)` | Computes the Mean Absolute Error (MAE) loss. |
| `ComputeMSEGradient(Tensor<>,Tensor<>)` | Computes the gradient of MSE loss for backpropagation. |
| `ComputeMSELoss(Tensor<>,Tensor<>)` | Computes the Mean Squared Error (MSE) loss. |
| `ComputeR2Score(Tensor<>,Tensor<>)` | Computes the R² (coefficient of determination) score. |
| `ComputeTotalLoss(Tensor<>,Tensor<>)` | Computes the total loss including sparsity regularization. |
| `Predict(Tensor<>)` | Performs prediction for regression. |
| `TrainStep(Tensor<>,Tensor<>,)` | Performs a single training step. |

