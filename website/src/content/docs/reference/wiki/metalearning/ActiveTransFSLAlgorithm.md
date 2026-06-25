---
title: "ActiveTransFSLAlgorithm<T, TInput, TOutput>"
description: "Implementation of ActiveTransFSL: Active Transductive Few-Shot Learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of ActiveTransFSL: Active Transductive Few-Shot Learning.

## How It Works

ActiveTransFSL performs standard inductive adaptation on support data, then applies
transductive refinement using query gradients. The refinement is focused on the most
uncertain parameter dimensions (measured by gradient magnitude), implementing an
active learning strategy in parameter space for transductive inference.

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

