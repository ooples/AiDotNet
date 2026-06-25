---
title: "OMLAlgorithm<T, TInput, TOutput>"
description: "Implementation of OML: Online Meta-Learning (Javed & White, 2019)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of OML: Online Meta-Learning (Javed & White, 2019).

## How It Works

OML partitions model parameters into a Representation Learning Network (RLN) and a
Prediction Learning Network (PLN). The RLN (first 1-f fraction of parameters) is only
updated in the outer loop, while the PLN (last f fraction) is adapted in the inner loop.
This division encourages the RLN to learn sparse, non-interfering representations that
support continual learning — the PLN can be quickly adapted without forgetting.

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

