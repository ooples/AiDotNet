---
title: "JMPAlgorithm<T, TInput, TOutput>"
description: "Implementation of JMP: Joint Multi-Phase meta-learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of JMP: Joint Multi-Phase meta-learning.

## How It Works

JMP divides the inner loop into two phases with distinct optimization characteristics.
Phase 1 (coarse) uses a higher learning rate for rapid, broad adaptation. Phase 2 (fine)
uses a lower learning rate with L2 regularization toward the Phase 1 result, enabling
careful refinement without deviating too far from the coarse solution.

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

