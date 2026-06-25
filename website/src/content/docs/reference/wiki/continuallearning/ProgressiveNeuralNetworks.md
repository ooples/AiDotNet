---
title: "ProgressiveNeuralNetworks<T>"
description: "Implements Progressive Neural Networks for continual learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ContinualLearning`

Implements Progressive Neural Networks for continual learning.

## For Beginners

Progressive Neural Networks prevent forgetting by freezing
previously learned networks and adding new "columns" (networks) for each new task.
The new columns can receive input from all previous columns through lateral connections,
enabling knowledge transfer without forgetting.

## How It Works

**How it works:**

**Architecture:**

**Advantages:**

**Disadvantages:**

**Reference:** Rusu, A.A. et al. "Progressive Neural Networks" (2016). arXiv.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ProgressiveNeuralNetworks(Boolean,Double)` | Initializes a new instance of the ProgressiveNeuralNetworks class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AccumulatesAcrossTasks` |  |
| `ColumnCount` | Gets the number of columns (tasks) in the progressive network. |
| `UseLateralConnections` | Gets whether lateral connections are enabled. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AfterTask(INeuralNetwork<>,ValueTuple<Tensor<>,Tensor<>>,Int32)` |  |
| `BeforeTask(INeuralNetwork<>,Int32)` |  |
| `ComputeLateralInput(List<Tensor<>>,List<Tensor<>>)` | Computes the lateral connection activation from previous columns. |
| `ComputeLoss(INeuralNetwork<>)` |  |
| `EstimateMemoryUsage` | Estimates the total memory usage of the progressive network. |
| `GetColumnParameters(Int32)` | Gets parameters for a specific task's column. |
| `GetFrozenParameterCount` | Gets the total number of frozen parameters across all completed tasks. |
| `GetNetworkStats` | Gets statistics about the progressive network structure. |
| `ModifyGradients(INeuralNetwork<>,Vector<>)` |  |
| `Reset` |  |

