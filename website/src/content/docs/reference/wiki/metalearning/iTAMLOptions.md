---
title: "iTAMLOptions<T, TInput, TOutput>"
description: "Configuration options for the iTAML (incremental Task-Agnostic Meta-Learning) algorithm."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for the iTAML (incremental Task-Agnostic Meta-Learning) algorithm.

## How It Works

iTAML (Rajasegaran et al., 2020) prevents catastrophic forgetting by maintaining an EMA
teacher model and applying knowledge distillation between teacher and student predictions.
Task-balanced gradient weighting normalizes gradient magnitudes across tasks to prevent
any single task from dominating the meta-update.

## Properties

| Property | Summary |
|:-----|:--------|
| `DistillationTemperature` | Temperature for softening predictions in the distillation loss. |
| `DistillationWeight` | Weight of the knowledge distillation loss between student and teacher predictions. |
| `TaskBalancingEnabled` | Whether to normalize gradient magnitudes across tasks to prevent high-loss tasks from dominating. |
| `TeacherEmaDecay` | EMA decay rate for the teacher model. |

