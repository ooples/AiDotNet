---
title: "SITrainer<T, TInput, TOutput>"
description: "Continual learning trainer using Synaptic Intelligence (SI)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ContinualLearning.Trainers`

Continual learning trainer using Synaptic Intelligence (SI).

## For Beginners

This trainer implements continual learning using SI,
which prevents catastrophic forgetting by tracking how much each parameter
contributes to reducing the loss during training.

## How It Works

**How SI Works:**

**Path Integral Formula:**

Where g_k is the gradient, Δθ_k is parameter change, and ξ is damping.

**Key Advantage:** SI computes importance online during training,
making it more computationally efficient than EWC which requires a separate pass.

**Usage Example:**

**Reference:** Zenke et al. "Continual Learning Through Synaptic Intelligence" (ICML 2017)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SITrainer(IFullModel<,,>,ILossFunction<>,IContinualLearnerConfig<>,IContinualLearningStrategy<,,>)` | Initializes a new SI trainer with default options. |
| `SITrainer(IFullModel<,,>,ILossFunction<>,IContinualLearnerConfig<>,IContinualLearningStrategy<,,>,SITrainerOptions<>)` | Initializes a new SI trainer with custom options. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SIStrategy` | Gets the SI-specific strategy if available. |

## Methods

| Method | Summary |
|:-----|:--------|
| `TrainOnTask(IDataset<,,>,IDataset<,,>,Int32)` |  |

