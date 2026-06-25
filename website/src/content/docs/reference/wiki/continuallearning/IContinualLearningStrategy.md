---
title: "IContinualLearningStrategy<T, TInput, TOutput>"
description: "Strategy interface for continual learning algorithms."
section: "API Reference"
---

`Interfaces` · `AiDotNet.ContinualLearning.Interfaces`

Strategy interface for continual learning algorithms.

## For Beginners

Different continual learning methods use different strategies
to prevent forgetting. This interface allows the trainer to work with any strategy.

## How It Works

**Strategy Types:**

- **Regularization-based:** Add penalty terms to protect important weights (EWC, SI, MAS)
- **Replay-based:** Store and replay old examples (Experience Replay, GEM)
- **Architecture-based:** Use separate parameters for different tasks (Progressive Networks, PackNet)
- **Distillation-based:** Use teacher model to preserve old knowledge (LwF)

**Reference:** De Lange et al. "A Continual Learning Survey: Defying Forgetting" (2021)

## Properties

| Property | Summary |
|:-----|:--------|
| `MemoryUsageBytes` | Gets the current memory usage of the strategy in bytes. |
| `ModifiesArchitecture` | Gets whether this strategy modifies the model architecture. |
| `Name` | Gets the name of the strategy. |
| `RequiresMemoryBuffer` | Gets whether this strategy requires storing examples from previous tasks. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AdjustGradients(Vector<>)` | Adjusts gradients to prevent forgetting. |
| `ComputeRegularizationLoss(IFullModel<,,>)` | Computes the regularization loss to prevent forgetting. |
| `FinalizeTask(IFullModel<,,>)` | Finalizes the task after training is complete. |
| `GetIncompatibilityReason(IFullModel<,,>)` | Gets a description of why the strategy is incompatible with a model. |
| `GetMetrics` | Gets strategy-specific metrics for monitoring. |
| `IsCompatibleWith(IFullModel<,,>)` | Validates that the strategy is compatible with the given model. |
| `Load(String)` | Loads the strategy state from a file. |
| `PrepareForTask(IFullModel<,,>,IDataset<,,>)` | Prepares the strategy for learning a new task. |
| `Reset` | Resets the strategy to its initial state. |
| `Save(String)` | Saves the strategy state to a file. |

