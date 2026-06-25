---
title: "LwFTrainer<T, TInput, TOutput>"
description: "Continual learning trainer using Learning without Forgetting (LwF)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ContinualLearning.Trainers`

Continual learning trainer using Learning without Forgetting (LwF).

## For Beginners

This trainer implements continual learning using LwF,
which prevents catastrophic forgetting through knowledge distillation from a
"teacher" model (the model before training on the new task).

## How It Works

**How LwF Works:**

**Key Advantage:** LwF doesn't need to store old task data - it only needs the
teacher model's outputs on current task data.

**Distillation Loss:**

Where T is the temperature parameter.

**Usage Example:**

**Reference:** Li and Hoiem "Learning without Forgetting" (2017)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LwFTrainer(IFullModel<,,>,ILossFunction<>,IContinualLearnerConfig<>,IContinualLearningStrategy<,,>)` | Initializes a new LwF trainer with default options. |
| `LwFTrainer(IFullModel<,,>,ILossFunction<>,IContinualLearnerConfig<>,IContinualLearningStrategy<,,>,LwFTrainerOptions<>)` | Initializes a new LwF trainer with custom options. |

## Properties

| Property | Summary |
|:-----|:--------|
| `LwFStrategy` | Gets the LwF-specific strategy if available. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ConvertToVector()` | Converts an output to a Vector for distillation loss computation. |
| `TrainOnTask(IDataset<,,>,IDataset<,,>,Int32)` |  |

