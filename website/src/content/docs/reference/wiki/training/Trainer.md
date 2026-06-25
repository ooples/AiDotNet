---
title: "Trainer<T>"
description: "Default trainer that delegates to the model's built-in `Train()` method each epoch."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Training`

Default trainer that delegates to the model's built-in `Train()` method each epoch.

## For Beginners

This is the standard trainer for models that know how to train
themselves (e.g., time series models like ARIMA and ExponentialSmoothing). Each epoch
it calls `model.Train(features, labels)`, then measures how well the model predicts
by computing the loss.

## How It Works

**Example usage from YAML:**

**Example usage with in-memory data:**

**Custom logging:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Trainer(String)` | Creates a trainer from a YAML configuration file. |
| `Trainer(TrainingRecipeConfig)` | Creates a trainer from a `TrainingRecipeConfig` object. |

## Methods

| Method | Summary |
|:-----|:--------|
| `TrainEpoch(Matrix<>,Vector<>,Int32)` | Trains the model for one epoch by calling its built-in Train method, then computes and returns the loss. |

