---
title: "DeepSurvOptions<T>"
description: "Configuration options for DeepSurv survival analysis model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for DeepSurv survival analysis model.

## For Beginners

Survival analysis is used when you want to predict "time until
an event happens" - like:

- How long until a machine fails?
- How long until a customer cancels their subscription?
- How long until a patient experiences disease recurrence?

DeepSurv uses a neural network to learn complex patterns in your data while properly
handling this censoring. It outputs a "risk score" - higher values mean higher risk
of the event happening sooner.

## How It Works

DeepSurv extends the Cox Proportional Hazards model using a deep neural network
to learn the relationship between covariates and survival outcomes. It optimizes
the negative partial log-likelihood of the Cox model.

## Properties

| Property | Summary |
|:-----|:--------|
| `Activation` | Gets or sets the activation function for hidden layers. |
| `BatchSize` | Gets or sets the batch size for training. |
| `DropoutRate` | Gets or sets the dropout rate for regularization. |
| `EarlyStoppingPatience` | Gets or sets the patience for early stopping. |
| `Epochs` | Gets or sets the number of training epochs. |
| `HiddenLayerSize` | Gets or sets the number of neurons in each hidden layer. |
| `L2Regularization` | Gets or sets the L2 regularization strength. |
| `LearningRate` | Gets or sets the learning rate for optimization. |
| `NumHiddenLayers` | Gets or sets the number of hidden layers. |
| `Seed` | Gets or sets the random seed for reproducibility. |
| `UseBatchNormalization` | Gets or sets whether to use batch normalization. |

