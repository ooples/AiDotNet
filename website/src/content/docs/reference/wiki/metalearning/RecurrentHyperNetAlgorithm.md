---
title: "RecurrentHyperNetAlgorithm<T, TInput, TOutput>"
description: "Implementation of Recurrent HyperNetwork for meta-learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of Recurrent HyperNetwork for meta-learning.

## How It Works

A GRU-like recurrent cell processes compressed gradient information at each adaptation
step, maintaining hidden state that captures the optimization trajectory. The recurrent
output is used to compute per-parameter learning rate modulation factors, enabling
adaptive step sizes that evolve through the inner loop based on gradient history.

**Algorithm:**

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` |  |
| `MetaTrain(TaskBatch<,,>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_gruWeights` | GRU weights: W_z, W_r, W_h — each (hidDim + inputDim) × hidDim. |

