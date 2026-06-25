---
title: "NeuralCVaROptions<T>"
description: "Configuration options for the NeuralCVaR model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the NeuralCVaR model.

## For Beginners

These options set the size of the neural network that
predicts CVaR (expected shortfall). More layers or larger layers increase
model capacity but can overfit if you don’t have enough data.

## Properties

| Property | Summary |
|:-----|:--------|
| `HiddenDimension` | Size of hidden layers. |
| `HiddenLayers` | Number of hidden layers in the network. |

