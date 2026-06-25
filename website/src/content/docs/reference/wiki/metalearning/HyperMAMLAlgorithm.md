---
title: "HyperMAMLAlgorithm<T, TInput, TOutput>"
description: "Implementation of HyperMAML (hypernetwork-based MAML initialization)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of HyperMAML (hypernetwork-based MAML initialization).

## For Beginners

HyperMAML improves MAML's starting point for each task:

**The problem with standard MAML:**
MAML learns ONE initialization that's supposed to be good for ALL tasks.
But some tasks are very different from each other, so one starting point
can't be optimal for everything.

**How HyperMAML fixes this:**

1. Look at the support set for the current task
2. Compute statistics about this specific task (means, variances, etc.)
3. Feed these statistics into a hypernetwork
4. The hypernetwork generates a CUSTOM initialization for this task
5. Then proceed with standard MAML inner-loop adaptation from this better start

**The benefit:**

- Task A (dog breeds): Gets an initialization tuned for fine-grained visual features
- Task B (vehicles vs animals): Gets an initialization tuned for coarse category differences
- Fewer inner-loop steps needed because the starting point is already close

## How It Works

HyperMAML uses a hypernetwork to generate task-specific initial parameters for MAML,
rather than using a single shared initialization. The hypernetwork conditions on
support set statistics to produce a better starting point for each task.

**Algorithm - HyperMAML:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HyperMAMLAlgorithm(HyperMAMLOptions<,,>)` | Initializes a new HyperMAML meta-learner. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` |  |
| `ComputeAuxLoss(TaskBatch<,,>)` | Computes the average loss over a task batch using the hypernetwork-generated initialization. |
| `GenerateTaskInit(Vector<>,Vector<>)` | Generates a task-specific initialization by running support features through the hypernetwork, then blends it with the shared initialization. |
| `InitializeHypernetwork` | Initializes the hypernetwork parameters. |
| `MetaTrain(TaskBatch<,,>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_hypernetParams` | Parameters for the initialization hypernetwork. |

