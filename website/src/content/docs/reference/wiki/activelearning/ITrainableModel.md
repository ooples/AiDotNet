---
title: "ITrainableModel<T, TInput, TOutput>"
description: "Interface for models that support training on datasets."
section: "API Reference"
---

`Interfaces` · `AiDotNet.ActiveLearning.Interfaces`

Interface for models that support training on datasets.

## For Beginners

A trainable model is one that can learn from data.
The training process adjusts the model's internal parameters to improve its
predictions based on the provided examples.

## How It Works

**Common Uses:**

## Properties

| Property | Summary |
|:-----|:--------|
| `IsTrained` | Gets whether the model has been trained. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Reset` | Resets the model to its initial untrained state. |
| `Train(IDataset<,,>)` | Trains the model on the provided dataset. |
| `Train(IDataset<,,>,Int32)` | Trains the model for a specified number of epochs. |

