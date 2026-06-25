---
title: "GradientEpisodicMemory<T, TInput, TOutput>"
description: "Gradient Episodic Memory (GEM) strategy for continual learning with gradient constraints."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ContinualLearning.Strategies`

Gradient Episodic Memory (GEM) strategy for continual learning with gradient constraints.

## For Beginners

GEM prevents forgetting by ensuring that parameter updates
for new tasks don't increase the loss on any previous task. It does this by:

1. Storing representative examples from each task (episodic memory)
2. Computing reference gradients for each previous task
3. Projecting the current gradient to satisfy all task constraints

## How It Works

**How Gradient Projection Works:**
If the current gradient g would increase loss on task k (i.e., g · g_k < 0), we project
g to the closest gradient g' such that g' · g_k ≥ 0 for all previous tasks.
This is a Quadratic Programming (QP) problem.

**GEM vs A-GEM:**

- **GEM:** Checks all task constraints individually. Strong guarantees but O(t) complexity.
- **A-GEM:** Uses average reference gradient. O(1) but weaker guarantees.

**Advantages:**

- Strong theoretical guarantees (never increases loss on previous tasks)
- Works with limited memory
- Applicable to any gradient-based optimizer

**Disadvantages:**

- Requires solving QP for gradient projection
- Needs access to gradients (not just loss values)
- Memory scales linearly with number of tasks

**References:**

- Lopez-Paz & Ranzato "Gradient Episodic Memory for Continual Learning" (NeurIPS 2017)
- Chaudhry et al. "Efficient Lifelong Learning with A-GEM" (ICLR 2019)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GradientEpisodicMemory(ILossFunction<>,GEMOptions<>)` | Initializes a new Gradient Episodic Memory strategy. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxExamples` |  |
| `MemoryBuffer` | Gets the episodic memory buffer. |
| `MemoryUsageBytes` |  |
| `ModifiesArchitecture` |  |
| `Name` |  |
| `RequiresMemoryBuffer` |  |
| `StoredExampleCount` |  |
| `StoredGradientCount` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AdjustGradients(Vector<>)` |  |
| `ClearMemory` |  |
| `ComputeAverageReferenceGradient` | Computes average gradient from a random sample of memory examples. |
| `ComputeRegularizationLoss(IFullModel<,,>)` |  |
| `FinalizeTask(IFullModel<,,>)` |  |
| `GetMetrics` |  |
| `GetStateForSerialization` |  |
| `LoadStateFromSerialization(Dictionary<String,JsonElement>)` |  |
| `PrepareForTask(IFullModel<,,>,IDataset<,,>)` |  |
| `ProjectGradient(Vector<>)` |  |
| `ProjectGradientAGEM(Vector<>)` | Projects gradient using A-GEM (average reference gradient). |
| `ProjectGradientGEM(Vector<>)` | Projects gradient using full GEM (all task constraints). |
| `Reset` |  |
| `SampleExamples(Int32)` |  |
| `SolveQPDual(Double[0:,0:],Double[])` | Solves the dual QP using projected gradient descent. |
| `StoreTaskExamples(IDataset<,,>)` |  |
| `StoreTaskGradient(Vector<>)` |  |
| `ViolatesConstraint(Vector<>)` |  |

