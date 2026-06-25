---
title: "GEMTrainer<T, TInput, TOutput>"
description: "Continual learning trainer using Gradient Episodic Memory (GEM)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ContinualLearning.Trainers`

Continual learning trainer using Gradient Episodic Memory (GEM).

## For Beginners

This trainer implements continual learning using GEM,
which prevents catastrophic forgetting by ensuring that gradient updates don't
increase the loss on any previous task.

## How It Works

**How GEM Works:**

**Gradient Projection Constraint:**

Where α_k are solved via quadratic programming.

**A-GEM Variant:** Uses only the average gradient of a random subset
of previous task examples, making it more efficient but potentially less effective.

**Usage Example:**

**References:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GEMTrainer(IFullModel<,,>,ILossFunction<>,IContinualLearnerConfig<>,IContinualLearningStrategy<,,>)` | Initializes a new GEM trainer with default options. |
| `GEMTrainer(IFullModel<,,>,ILossFunction<>,IContinualLearnerConfig<>,IContinualLearningStrategy<,,>,GEMTrainerOptions<>)` | Initializes a new GEM trainer with custom options. |

## Properties

| Property | Summary |
|:-----|:--------|
| `GEMStrategy` | Gets the GEM-specific strategy if available. |

## Methods

| Method | Summary |
|:-----|:--------|
| `TrainOnTask(IDataset<,,>,IDataset<,,>,Int32)` |  |

