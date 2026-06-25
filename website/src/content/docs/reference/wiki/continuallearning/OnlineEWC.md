---
title: "OnlineEWC<T>"
description: "Implements Online Elastic Weight Consolidation (Online EWC) for continual learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ContinualLearning`

Implements Online Elastic Weight Consolidation (Online EWC) for continual learning.

## For Beginners

Online EWC is a memory-efficient variant of EWC that maintains
a single running approximation of the Fisher Information Matrix across all tasks, rather
than storing separate matrices for each task.

## How It Works

**How it works:**

**Formula:**

F̃ = γ * F_old + F_new

θ̃* = (γ * F_old * θ*_old + F_new * θ_new) / (γ * F_old + F_new)

**Advantages:**

**Reference:** Schwarz, J. et al. "Progress & Compress: A scalable framework for
continual learning" (2018). ICML.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OnlineEWC(Double,Double)` | Initializes a new instance of the OnlineEWC class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AccumulatesAcrossTasks` |  |
| `Gamma` | Gets the decay factor for old Fisher information. |
| `TaskCount` | Gets the number of tasks processed. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AfterTask(INeuralNetwork<>,ValueTuple<Tensor<>,Tensor<>>,Int32)` |  |
| `BeforeTask(INeuralNetwork<>,Int32)` |  |
| `ComputeLogLikelihoodGradient(Tensor<>,Tensor<>)` | Computes the gradient of the log-likelihood. |
| `ComputeLoss(INeuralNetwork<>)` |  |
| `ExtractSample(Tensor<>,Int32)` | Extracts a single sample from a batch tensor. |
| `MergeFisherAndParameters(Vector<>,Vector<>)` | Merges new Fisher information and parameters with the running estimate. |
| `ModifyGradients(INeuralNetwork<>,Vector<>)` |  |
| `Reset` |  |

