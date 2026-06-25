---
title: "MetaFDMixupAlgorithm<T, TInput, TOutput>"
description: "Implementation of Meta-FDMixup: Feature-Distribution Mixup for cross-domain few-shot learning (Xu et al., CVPR 2021)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of Meta-FDMixup: Feature-Distribution Mixup for cross-domain
few-shot learning (Xu et al., CVPR 2021).

## How It Works

Meta-FDMixup performs gradient-level mixup between tasks in a meta-batch to improve
cross-domain robustness. For each task, with probability p, its inner-loop gradient
is mixed with a randomly selected other task's gradient using a Beta-distributed
coefficient λ. The outer loop also applies feature distribution alignment by
penalizing the variance of per-task gradient directions.

**Algorithm:**

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` |  |
| `ComputeGradientAlignmentPenalty(List<Vector<>>)` | Computes variance of gradient directions as alignment penalty. |
| `MetaTrain(TaskBatch<,,>)` |  |

