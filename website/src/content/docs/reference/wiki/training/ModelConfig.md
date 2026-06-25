---
title: "ModelConfig"
description: "Configuration for the model section of a training recipe."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Training.Configuration`

Configuration for the model section of a training recipe.

## For Beginners

This defines which model to use and its parameters.
The name should match a `TimeSeriesModelType` value
(e.g., "ARIMA", "ExponentialSmoothing", "SARIMA").

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` | Gets or sets the name of the model type to create. |
| `Params` | Gets or sets the model-specific parameters as key-value pairs. |

