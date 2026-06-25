---
title: "LossFunctionConfig"
description: "Configuration for the loss function section of a training recipe."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Training.Configuration`

Configuration for the loss function section of a training recipe.

## For Beginners

The loss function measures how far the model's predictions are from
the correct answers. The name should match a `LossType` value
(e.g., "MeanSquaredError", "CrossEntropy", "Huber").

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` | Gets or sets the name of the loss function type to create. |
| `Params` | Gets or sets loss function-specific parameters as key-value pairs. |

