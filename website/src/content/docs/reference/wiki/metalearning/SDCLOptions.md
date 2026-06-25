---
title: "SDCLOptions<T, TInput, TOutput>"
description: "Configuration options for SDCL: Self-Distillation Collaborative Learning for meta-learning."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for SDCL: Self-Distillation Collaborative Learning for meta-learning.

## How It Works

SDCL applies self-distillation within the meta-learning framework. A teacher model
(exponential moving average of the student) provides soft targets that regularize
the adapted student's predictions. The KL divergence between teacher and student
predictions acts as a collaborative learning signal that stabilizes adaptation
and improves cross-domain generalization.

## Properties

| Property | Summary |
|:-----|:--------|
| `DistillationTemperature` | Temperature for softening teacher/student predictions. |
| `DistillationWeight` | Weight for the distillation (KL divergence) loss. |
| `TeacherEmaDecay` | EMA decay rate for the teacher model. |

