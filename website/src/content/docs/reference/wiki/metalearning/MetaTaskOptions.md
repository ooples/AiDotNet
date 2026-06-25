---
title: "MetaTaskOptions<T, TInput, TOutput>"
description: "Configuration options for the MetaTask (Meta-learned Task Augmentation) algorithm."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for the MetaTask (Meta-learned Task Augmentation) algorithm.

## How It Works

MetaTask generates synthetic tasks by interpolating gradients between pairs of real tasks
using Beta-distributed mixing coefficients. The synthetic tasks augment the task distribution,
improving generalization of the meta-learned initialization.

## Properties

| Property | Summary |
|:-----|:--------|
| `InterpolationAlpha` | Alpha parameter for Beta(α,α) distribution used to sample interpolation coefficients. |
| `NumSyntheticTasks` | Number of synthetic (interpolated) tasks generated per meta-batch. |
| `SyntheticWeight` | Weight on the synthetic task losses relative to real task losses. |

