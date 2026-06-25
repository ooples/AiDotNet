---
title: "ExperienceReplay<T>"
description: "Implements Experience Replay for continual learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ContinualLearning`

Implements Experience Replay for continual learning.

## For Beginners

Experience Replay is one of the simplest and most effective
continual learning strategies. It stores examples from previous tasks and mixes them with
new task data during training, directly rehearsing old knowledge.

## How It Works

**How it works:**

**Buffer Strategies:**

**Advantages:**

**Reference:** Ratcliff, R. "Connectionist models of recognition memory" (1990).
Psychological Review. (Original concept); Rolnick et al. "Experience Replay for Continual
Learning" (2019). NeurIPS. (Modern application)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ExperienceReplay(Int32,Double,ExperienceReplay<>.BufferStrategy,Double,Nullable<Int32>)` | Initializes a new instance of the ExperienceReplay class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AccumulatesAcrossTasks` |  |
| `BufferSize` | Gets the current buffer size. |
| `ReplayRatio` | Gets the replay ratio. |
| `Strategy` | Gets the buffer strategy. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddClassBalanced(Tensor<>,Tensor<>,Int32)` | Adds sample maintaining class balance. |
| `AddReservoir(Tensor<>,Tensor<>,Int32)` | Adds sample using reservoir sampling. |
| `AddRing(Tensor<>,Tensor<>,Int32)` | Adds sample using ring buffer (FIFO). |
| `AddToBuffer(Tensor<>,Tensor<>,Int32)` | Adds samples from a task to the replay buffer. |
| `AfterTask(INeuralNetwork<>,ValueTuple<Tensor<>,Tensor<>>,Int32)` |  |
| `BeforeTask(INeuralNetwork<>,Int32)` |  |
| `BuildBatchFromIndices(List<Int32>)` | Builds a batch from buffer indices. |
| `CombineTensors(List<Tensor<>>,List<Tensor<>>)` | Combines multiple tensors into single batch tensors. |
| `ComputeLoss(INeuralNetwork<>)` |  |
| `ExtractSample(Tensor<>,Int32)` | Extracts a single sample from a batch tensor. |
| `ModifyGradients(INeuralNetwork<>,Vector<>)` |  |
| `Reset` |  |
| `SampleMixedBatch(Tensor<>,Tensor<>,Int32)` | Samples a mixed batch combining current task data with replay data. |
| `SampleReplayBatch` | Samples a batch from the replay buffer for training. |

