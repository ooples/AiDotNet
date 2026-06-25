---
title: "ITrainer<T>"
description: "Interface for training machine learning models from configuration-driven recipes."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for training machine learning models from configuration-driven recipes.

## For Beginners

A trainer takes a training recipe (configuration) and runs the full
training process: loading data, creating the model, running epochs, and returning results.
This is the "run my experiment" interface.

## Properties

| Property | Summary |
|:-----|:--------|
| `Config` | Gets the training recipe configuration used by this trainer. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Run` | Runs the full training loop and returns the result. |
| `RunAsync` | Runs the full training loop asynchronously and returns the result. |

