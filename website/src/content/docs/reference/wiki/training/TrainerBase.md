---
title: "TrainerBase<T>"
description: "Abstract base class for all trainers, providing shared infrastructure for configuration-driven training pipelines."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Training`

Abstract base class for all trainers, providing shared infrastructure for
configuration-driven training pipelines.

## For Beginners

Think of TrainerBase as a template for running experiments.
It handles all the boilerplate—loading data, setting up the model/optimizer/loss,
timing the run, and collecting results—so that concrete trainers only need to
implement the actual training strategy in `Int32)`.

## How It Works

The inheritance pattern follows the project's architecture requirements:
`ITrainer<T>` (interface) → `TrainerBase<T>` (base) → `Trainer<T>` (concrete).
To create a custom training strategy, inherit from this base class and override
`Int32)`.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TrainerBase(String)` | Creates a trainer base from a YAML configuration file. |
| `TrainerBase(TrainingRecipeConfig)` | Creates a trainer base from a `TrainingRecipeConfig` object. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Config` |  |
| `LogAction` | Gets or sets the action used for logging training messages. |
| `LossFunction` | Gets the loss function created from the configuration. |
| `Model` | Gets the model created from the configuration. |
| `Optimizer` | Gets the optimizer created from the configuration, if one was specified. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ResolveData` | Resolves the feature/label data from either in-memory data or the CSV loader. |
| `Run` |  |
| `RunAsync` |  |
| `SetData(Matrix<>,Vector<>)` | Sets in-memory feature and label data for training, bypassing CSV loading. |
| `TrainEpoch(Matrix<>,Vector<>,Int32)` | Executes a single training epoch and returns the computed loss. |

