---
title: "PipelineOperationType"
description: "Types of pipeline operations."
section: "API Reference"
---

`Enums` · `AiDotNet.Interfaces`

Types of pipeline operations.

## How It Works

Traditional schedules (GPipe, 1F1B) use Forward and Backward.
Zero Bubble schedules decompose Backward into BackwardInput + BackwardWeight
to enable filling pipeline bubbles with weight gradient computation.

**Reference:** Qi et al., "Zero Bubble Pipeline Parallelism", ICLR 2024.
https://arxiv.org/abs/2401.10241

## Fields

| Field | Summary |
|:-----|:--------|
| `Backward` | Combined backward pass (gradient computation) through the stage's layers. |
| `BackwardInput` | Backward pass computing only activation gradients (dL/dInput). |
| `BackwardWeight` | Backward pass computing only weight gradients (dL/dWeights). |
| `Forward` | Forward pass through the stage's layers. |

