---
title: "SynapticIntelligence<T>"
description: "Implements Synaptic Intelligence (SI) for continual learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ContinualLearning`

Implements Synaptic Intelligence (SI) for continual learning.

## For Beginners

Synaptic Intelligence is similar to EWC but estimates weight
importance online during training rather than computing Fisher information after training.
It tracks how much each weight contributes to the loss reduction during learning.

## How It Works

**How it works:**

**Formula:** Ω_i = Σ_tasks [ω_i^task / (Δθ_i^task)² + ξ]

where ω_i is the path integral of gradients for weight i.

**Advantages over EWC:**

**Reference:** Zenke, F., Poole, B., and Ganguli, S. "Continual Learning Through
Synaptic Intelligence" (2017). ICML.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SynapticIntelligence(Double,Double)` | Initializes a new instance of the SynapticIntelligence class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AccumulatesAcrossTasks` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AfterTask(INeuralNetwork<>,ValueTuple<Tensor<>,Tensor<>>,Int32)` |  |
| `BeforeTask(INeuralNetwork<>,Int32)` |  |
| `ComputeLoss(INeuralNetwork<>)` |  |
| `HasPreviousTasks` | Checks if there are any previous tasks stored. |
| `ModifyGradients(INeuralNetwork<>,Vector<>)` |  |
| `Reset` |  |
| `UpdatePathIntegral(Vector<>,Vector<>)` | Updates the path integral with the current gradient and parameter change. |

