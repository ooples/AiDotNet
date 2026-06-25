---
title: "MetaContinualALAlgorithm<T, TInput, TOutput>"
description: "Implementation of MetaContinualAL: Meta-Continual Active Learning with uncertainty-guided parameter-selective adaptation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of MetaContinualAL: Meta-Continual Active Learning with uncertainty-guided
parameter-selective adaptation.

## How It Works

MetaContinualAL uses gradient-norm-based uncertainty estimation to identify the most
informative parameter dimensions and focuses adaptation effort there. A running EMA
calibration tracks mean/variance of per-parameter gradient magnitudes. Parameters with
above-average uncertainty receive amplified learning rates, while low-uncertainty
parameters are dampened — similar to active learning's acquisition function but applied
in parameter space.

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
| `LowUncertaintyDampening` | Learning rate dampening factor for low-uncertainty parameters. |
| `_uncertaintyMean` | Running mean of per-parameter gradient magnitudes. |
| `_uncertaintyVar` | Running variance of per-parameter gradient magnitudes. |

