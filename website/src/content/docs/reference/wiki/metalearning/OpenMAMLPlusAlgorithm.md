---
title: "OpenMAMLPlusAlgorithm<T, TInput, TOutput>"
description: "Implementation of Open-MAML++: MAML with per-parameter learning rates and open-set novelty detection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of Open-MAML++: MAML with per-parameter learning rates and
open-set novelty detection.

## How It Works

Open-MAML++ combines MAML++ improvements (per-parameter learning rates, multi-step
loss, learned layer-specific LR) with open-set recognition. The algorithm meta-learns
a novelty threshold based on prediction entropy: samples with entropy exceeding the
threshold are classified as novel/unknown. The multi-step loss ensures stable
intermediate adaptation steps.

**Algorithm:**

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` |  |
| `ComputePredictionEntropy(Vector<>)` | Computes prediction entropy: H(p) = -Σ p_i log(p_i). |
| `MetaTrain(TaskBatch<,,>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_noveltyThreshold` | Meta-learned novelty threshold (on prediction entropy). |
| `_perParamLR` | Meta-learned per-parameter learning rates (MAML++ style). |

