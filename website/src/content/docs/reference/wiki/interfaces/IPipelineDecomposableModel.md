---
title: "IPipelineDecomposableModel<T, TInput, TOutput>"
description: "Interface for models that support decomposing the backward pass into separate activation gradient and weight gradient computations."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for models that support decomposing the backward pass into separate
activation gradient and weight gradient computations. This enables Zero Bubble
pipeline schedules (ZB-H1, ZB-H2, ZB-V) to overlap weight gradient computation
with other pipeline stages.

## For Beginners

Most models compute all gradients at once. This interface lets
advanced pipeline schedules split that work into two parts: one that's urgent (the upstream
stage is waiting for it) and one that can wait (filling idle time in the pipeline).

If your model doesn't implement this interface, pipeline schedules will automatically
fall back to computing both gradient types together (which still works, just can't
fill bubbles as effectively).

## How It Works

Standard backward passes compute both dL/dInput (activation gradients) and dL/dWeights
(weight gradients) together. This interface allows splitting them:

**Reference:** Qi et al., "Zero Bubble Pipeline Parallelism", ICLR 2024 Spotlight.
https://arxiv.org/abs/2401.10241

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeActivationGradients(,)` | Computes only the activation gradients (dL/dInput) for the backward pass. |
| `ComputeWeightGradients(,,Object)` | Computes only the weight gradients (dL/dWeights) for the backward pass. |

