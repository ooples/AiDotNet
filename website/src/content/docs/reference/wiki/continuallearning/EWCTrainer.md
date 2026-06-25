---
title: "EWCTrainer<T, TInput, TOutput>"
description: "Continual learning trainer using Elastic Weight Consolidation (EWC)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ContinualLearning.Trainers`

Continual learning trainer using Elastic Weight Consolidation (EWC).

## For Beginners

This trainer implements continual learning using EWC,
which prevents catastrophic forgetting by protecting important parameters from previous tasks.

## How It Works

**How EWC Works:**

**Online vs Offline EWC:**

**Usage Example:**

**Reference:** Kirkpatrick et al. "Overcoming catastrophic forgetting in neural networks" (PNAS 2017)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EWCTrainer(IFullModel<,,>,ILossFunction<>,IContinualLearnerConfig<>,IContinualLearningStrategy<,,>)` | Initializes a new EWC trainer with default options. |
| `EWCTrainer(IFullModel<,,>,ILossFunction<>,IContinualLearnerConfig<>,IContinualLearningStrategy<,,>,EWCTrainerOptions<>)` | Initializes a new EWC trainer with custom options. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EWCStrategy` | Gets the EWC-specific strategy if available. |

## Methods

| Method | Summary |
|:-----|:--------|
| `TrainOnTask(IDataset<,,>,IDataset<,,>,Int32)` |  |

