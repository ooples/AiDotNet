---
title: "PackNet<T, TInput, TOutput>"
description: "PackNet strategy for continual learning through network pruning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ContinualLearning.Strategies`

PackNet strategy for continual learning through network pruning.

## For Beginners

PackNet "packs" multiple tasks into a single network
by pruning unimportant parameters after each task and using the freed capacity
for new tasks. It's like having multiple neural networks compressed into one.

## How It Works

**How it works:**

**The Math:**

After training on task t:

1. Compute importance: I_i = |θ_i| (magnitude-based) or I_i = |∂L/∂θ_i| * |θ_i|

2. Prune: mask_t = I > percentile(I, prune_ratio * 100)

3. Freeze: θ_frozen = θ * mask_t

4. Free capacity: available = (1 - mask_t)

**Comparison to Other Methods:**

PackNet has O(1) memory per task but is limited by network capacity.
Best for scenarios where network size is constrained and tasks can share features.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PackNet(ILossFunction<>,Double)` | Initializes a new PackNet strategy with default options. |
| `PackNet(ILossFunction<>,PackNetOptions<>)` | Initializes a new PackNet strategy with custom options. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AvailableCapacity` | Gets the available parameter capacity (fraction of parameters not yet assigned). |
| `MemoryUsageBytes` |  |
| `ModifiesArchitecture` |  |
| `Name` |  |
| `PruningRatio` | Gets the pruning ratio. |
| `RequiresMemoryBuffer` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AdjustGradients(Vector<>)` |  |
| `ApplyTaskMask(Vector<>,Int32)` | Applies the task mask to parameters for inference on a specific task. |
| `ComputeImportanceScores(Vector<>)` | Computes importance scores for each parameter. |
| `ComputeRegularizationLoss(IFullModel<,,>)` |  |
| `ComputeTaskMask(Vector<>,Vector<>)` | Computes the mask for which parameters to keep for this task. |
| `FinalizeTask(IFullModel<,,>)` |  |
| `GetParameterCountByTask` | Gets the count of parameters assigned to each task. |
| `GetParameterOwnership` | Gets ownership information for all parameters. |
| `GetStateForSerialization` |  |
| `GetTaskMask(Int32)` | Gets the mask for a specific task (which parameters to use during inference). |
| `PrepareForTask(IFullModel<,,>,IDataset<,,>)` |  |
| `Reset` |  |
| `UpdateParameterOwnership(Boolean[],Int32)` | Updates parameter ownership based on the task mask. |

