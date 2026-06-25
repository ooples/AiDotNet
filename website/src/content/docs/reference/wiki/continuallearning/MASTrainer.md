---
title: "MASTrainer<T, TInput, TOutput>"
description: "Continual learning trainer using Memory Aware Synapses (MAS)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ContinualLearning.Trainers`

Continual learning trainer using Memory Aware Synapses (MAS).

## For Beginners

This trainer implements continual learning using MAS,
which prevents catastrophic forgetting by measuring how sensitive the network
output is to each parameter. The key advantage is that MAS is unsupervised -
it doesn't need task labels, just input data!

## How It Works

**How MAS Works:**

**MAS Regularization:**

Where Ω_i is the accumulated importance and θ*_i are the optimal parameters.

**Key Advantages over EWC:**

**Usage Example:**

**Reference:** Aljundi et al. "Memory Aware Synapses: Learning what (not) to forget" (ECCV 2018)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MASTrainer(IFullModel<,,>,ILossFunction<>,IContinualLearnerConfig<>,IContinualLearningStrategy<,,>)` | Initializes a new MAS trainer with default options. |
| `MASTrainer(IFullModel<,,>,ILossFunction<>,IContinualLearnerConfig<>,IContinualLearningStrategy<,,>,MASTrainerOptions<>)` | Initializes a new MAS trainer with custom options. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MASStrategy` | Gets the MAS-specific strategy if available. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeOutputNorm()` | Computes the L2 norm of an output for sensitivity tracking. |
| `ConvertToVector()` | Converts an output to a Vector for norm computation. |
| `TrainOnTask(IDataset<,,>,IDataset<,,>,Int32)` |  |

