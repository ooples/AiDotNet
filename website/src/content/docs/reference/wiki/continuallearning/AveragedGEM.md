---
title: "AveragedGEM<T>"
description: "Implements Averaged Gradient Episodic Memory (A-GEM) for continual learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ContinualLearning`

Implements Averaged Gradient Episodic Memory (A-GEM) for continual learning.

## For Beginners

A-GEM is a more efficient version of GEM that uses a single
random sample from the episodic memory instead of checking constraints against all stored
examples. This makes it much faster while maintaining similar performance.

## How It Works

**How it works:**

**Key Difference from GEM:**

**Projection Formula:**

If g · g_ref < 0: g_proj = g - (g · g_ref / g_ref · g_ref) × g_ref

**Reference:** Chaudhry, A. et al. "Efficient Lifelong Learning with A-GEM" (2019). ICLR.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AveragedGEM(Int32,Int32,Double,Nullable<Int32>)` | Initializes a new instance of the AveragedGEM class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AccumulatesAcrossTasks` |  |
| `TaskCount` | Gets the number of tasks stored in episodic memory. |
| `TotalMemorySize` | Gets the total number of samples in episodic memory. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AfterTask(INeuralNetwork<>,ValueTuple<Tensor<>,Tensor<>>,Int32)` |  |
| `BeforeTask(INeuralNetwork<>,Int32)` |  |
| `ComputeLoss(INeuralNetwork<>)` |  |
| `ComputeLossGradient(Tensor<>,Tensor<>)` | Computes the gradient of the loss. |
| `ModifyGradients(INeuralNetwork<>,Vector<>)` |  |
| `Reset` |  |
| `SampleFromEpisodicMemory` | Samples a random batch from all episodic memories combined. |
| `SampleMemory(Tensor<>,Tensor<>)` | Samples examples from task data to store in memory. |
| `SampleTensor(Tensor<>,Int32[])` | Samples specific indices from a tensor. |

