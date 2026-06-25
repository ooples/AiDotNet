---
title: "IIntermediateActivationStrategy<T>"
description: "Defines methods for distillation strategies that utilize intermediate layer activations."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines methods for distillation strategies that utilize intermediate layer activations.

## For Beginners

Some advanced distillation strategies don't just compare final outputs.
They also compare what's happening inside the models at intermediate layers. This interface is for
those advanced strategies.

## How It Works

**Example Strategies Needing Intermediate Activations:**

- **Feature-Based Distillation (FitNets)**: Match intermediate layer features between teacher and student
- **Attention Transfer**: Transfer attention patterns from internal layers
- **Neuron Selectivity**: Match how individual neurons respond across batches
- **Relational Knowledge Distillation**: Transfer relationships between layer activations

**Why Separate Interface?**
Not all strategies need intermediate activations. Simple response-based distillation only needs final outputs.
This interface is optional - only implement it if your strategy needs access to internal layer outputs.

**Usage Pattern:**

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeIntermediateGradient(IntermediateActivations<>,IntermediateActivations<>)` | Computes gradients of the intermediate activation loss with respect to student activations. |
| `ComputeIntermediateLoss(IntermediateActivations<>,IntermediateActivations<>)` | Computes a loss component based on intermediate layer activations for a batch. |

