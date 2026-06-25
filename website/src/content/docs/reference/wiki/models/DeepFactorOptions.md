---
title: "DeepFactorOptions<T>"
description: "Configuration options for the DeepFactor (Deep Factor Model) for time series forecasting."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the DeepFactor (Deep Factor Model) for time series forecasting.

## For Beginners

DeepFactor is designed for forecasting many related time series:

**What is Factor Modeling?**
Factor models assume observed variables are driven by hidden "factors":

- Global factors: Market-wide patterns (economy, weather, trends)
- Factor loadings: How much each series is affected by each factor
- Local component: Series-specific noise and behavior

**Example - Retail Sales:**
Factors might represent:

- F1: Overall economic conditions (affects all stores)
- F2: Holiday shopping season (affects all stores differently)
- F3: Regional weather (affects nearby stores similarly)

Each store's sales = (loading1 * F1) + (loading2 * F2) + (loading3 * F3) + local

**Why "Deep" Factor?**
Traditional factor models use linear relationships.
DeepFactor uses neural networks to:

- Learn non-linear factor dynamics
- Automatically discover factor structure
- Capture complex cross-series dependencies

**Benefits:**

- Captures shared patterns across many time series efficiently
- Reduces overfitting when series are related
- Works well for hierarchical forecasting (stores in regions, products in categories)
- Interpretable through factor analysis

## How It Works

DeepFactor combines factor modeling with deep learning for multivariate time series.
It decomposes time series into global factors (shared patterns) and local components
(series-specific behavior), learning both through neural networks.

**Reference:** Wang et al., "Deep Factors for Forecasting", 2019.
https://arxiv.org/abs/1905.12417

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DeepFactorOptions` | Initializes a new instance of the `DeepFactorOptions` class with default values. |
| `DeepFactorOptions(DeepFactorOptions<>)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the batch size for training. |
| `DropoutRate` | Gets or sets the dropout rate for regularization. |
| `Epochs` | Gets or sets the number of training epochs. |
| `FactorHiddenDimension` | Gets or sets the hidden dimension for the factor model RNN. |
| `ForecastHorizon` | Gets or sets the forecast horizon (prediction length). |
| `LearningRate` | Gets or sets the learning rate for training. |
| `LocalHiddenDimension` | Gets or sets the hidden dimension for the local model. |
| `LookbackWindow` | Gets or sets the lookback window size (input sequence length). |
| `NumFactorLayers` | Gets or sets the number of layers in the factor model. |
| `NumFactors` | Gets or sets the number of latent factors. |
| `NumLocalLayers` | Gets or sets the number of layers in the local model. |

