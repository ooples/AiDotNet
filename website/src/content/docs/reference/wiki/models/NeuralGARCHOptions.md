---
title: "NeuralGARCHOptions<T>"
description: "Configuration options for the Neural GARCH volatility model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the Neural GARCH volatility model.

## For Beginners

These settings control how much history the model sees,
how big the neural network is, and how far ahead you want to forecast volatility.

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate for regularization. |
| `ForecastHorizon` | Gets or sets the forecast horizon (number of future steps). |
| `HiddenSize` | Gets or sets the hidden layer width. |
| `LookbackWindow` | Gets or sets the lookback window (number of past time steps). |
| `NumAssets` | Gets or sets the number of assets modeled at once. |
| `NumLayers` | Gets or sets the number of hidden layers. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates the option values. |

