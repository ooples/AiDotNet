---
title: "OptimizerConfig"
description: "Configuration for the optimizer section of a training recipe."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Training.Configuration`

Configuration for the optimizer section of a training recipe.

## For Beginners

The optimizer controls how the model learns from its mistakes.
The name should match an `OptimizerType` value
(e.g., "Adam", "GradientDescent", "Normal").

## Properties

| Property | Summary |
|:-----|:--------|
| `LearningRate` | Gets or sets the learning rate for the optimizer. |
| `Name` | Gets or sets the name of the optimizer type to create. |
| `Params` | Gets or sets additional optimizer parameters as key-value pairs. |

