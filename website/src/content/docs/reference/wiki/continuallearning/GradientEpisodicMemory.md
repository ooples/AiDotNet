---
title: "GradientEpisodicMemory<T>"
description: "Implements Gradient Episodic Memory (GEM) for continual learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ContinualLearning`

Implements Gradient Episodic Memory (GEM) for continual learning.

## For Beginners

Gradient Episodic Memory is like having a safety net that
catches you if you're about to forget something important. It stores examples from
previous tasks and checks each gradient update to make sure it won't hurt performance
on those stored examples.

## How It Works

**How it works:**

**Reference:** Lopez-Paz and Ranzato, "Gradient Episodic Memory for Continual
Learning" (2017). NeurIPS.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GradientEpisodicMemory(Int32,Double,Double)` | Initializes a new instance of the GradientEpisodicMemory class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AccumulatesAcrossTasks` |  |
| `Margin` | Gets or sets the margin for the gradient constraint. |
| `TaskCount` | Gets the number of tasks currently stored in memory. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AfterTask(INeuralNetwork<>,ValueTuple<Tensor<>,Tensor<>>,Int32)` |  |
| `BeforeTask(INeuralNetwork<>,Int32)` |  |
| `ComputeLoss(INeuralNetwork<>)` |  |
| `ComputeLossGradient(Tensor<>,Tensor<>)` | Computes the gradient of the loss with respect to output. |
| `ModifyGradients(INeuralNetwork<>,Vector<>)` |  |
| `ProjectGradient(Vector<>,List<Int32>)` | Projects the gradient to satisfy constraints using a simplified QP solver. |
| `Reset` |  |
| `SampleMemory(Tensor<>,Tensor<>)` | Samples examples from task data to store in episodic memory. |
| `SampleTensor(Tensor<>,Int32[])` | Samples specific indices from a tensor. |
| `UpdateReferenceGradients(INeuralNetwork<>)` | Updates reference gradients for all stored tasks. |

