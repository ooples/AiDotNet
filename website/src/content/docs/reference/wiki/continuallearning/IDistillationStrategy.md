---
title: "IDistillationStrategy<T, TInput, TOutput>"
description: "Extended strategy interface for knowledge distillation-based strategies."
section: "API Reference"
---

`Interfaces` · `AiDotNet.ContinualLearning.Interfaces`

Extended strategy interface for knowledge distillation-based strategies.

## Properties

| Property | Summary |
|:-----|:--------|
| `DistillationWeight` | Gets the weight for distillation loss vs task loss. |
| `TeacherModel` | Gets the teacher model (frozen copy from before current task). |
| `Temperature` | Gets the distillation temperature. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeDistillationLoss(,)` | Computes the distillation loss between teacher and student outputs. |

