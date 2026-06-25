---
title: "ExperienceReplayBuffer<T, TInput, TOutput>"
description: "A memory buffer for storing examples from previous tasks for experience replay."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ContinualLearning.Memory`

A memory buffer for storing examples from previous tasks for experience replay.

## For Beginners

Experience replay stores a small number of examples from
previous tasks and intermixes them with new task data during training. This helps
prevent catastrophic forgetting by reminding the model of what it learned before.

## How It Works

**Why It Works:** When a neural network learns a new task, it adjusts its weights
to minimize error on that task. Without replay, these adjustments can increase error on
previous tasks. By mixing old examples with new training data, the network must maintain
performance on both old and new tasks.

**Memory Management:** Since storing all examples is impractical, we use smart
sampling strategies to select the most representative examples. Different strategies
work better for different scenarios:

- **Reservoir:** Fair representation, good for general use
- **ClassBalanced:** Equal class representation, best for imbalanced data
- **Herding:** Class-mean exemplars, from iCaRL paper

**References:**

- Chaudhry et al. "Continual Learning with Tiny Episodic Memories" (2019)
- Rebuffi et al. "iCaRL: Incremental Classifier and Representation Learning" (2017)
- Rolnick et al. "Experience Replay for Continual Learning" (2019)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ExperienceReplayBuffer(Int32,MemorySamplingStrategy,ReplaySamplingStrategy,Nullable<Int32>)` | Initializes a new experience replay buffer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AddStrategy` | Gets the sampling strategy used when adding examples. |
| `Count` | Gets the current number of stored examples. |
| `DefaultLossFunction` |  |
| `EstimatedMemoryBytes` | Gets the estimated memory usage in bytes. |
| `IsFull` | Gets whether the buffer is at capacity. |
| `MaxSize` | Gets the maximum capacity of the buffer. |
| `ReplayStrategy` | Gets the sampling strategy used during replay. |
| `TaskCount` | Gets the number of distinct tasks represented in the buffer. |
| `TotalReplaySamples` | Gets the total number of samples returned via replay. |
| `TotalSamplesProcessed` | Gets the total number of samples processed (added) since creation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddTaskExamples(IDataset<,,>,Int32,Nullable<Int32>)` | Adds examples from a task to the buffer using the configured sampling strategy. |
| `Clear` | Clears all stored examples. |
| `ComputeSquaredDistance(Double[],Double[])` | Computes the squared Euclidean distance between two feature vectors. |
| `DeepCopy` |  |
| `ExtractFeatures()` | Extracts feature values from an input for distance computation. |
| `GetAll` | Gets all stored examples. |
| `GetParameters` |  |
| `GetStatistics` | Gets statistics about the buffer. |
| `GetTaskCounts` | Gets the count of examples per task. |
| `GetTaskExamples(Int32)` | Gets all examples for a specific task. |
| `Predict()` |  |
| `RemoveTask(Int32)` | Removes all examples from a specific task. |
| `SampleBatch(Int32)` | Samples a batch of examples from the buffer using the configured replay strategy. |
| `SampleFromTask(Int32,Int32)` | Samples examples from a specific task. |
| `SetParameters(Vector<>)` |  |
| `Train(,)` |  |
| `UpdatePriority(Int32,Double)` | Updates the priority of a sample (for priority-based replay). |
| `WithParameters(Vector<>)` |  |

