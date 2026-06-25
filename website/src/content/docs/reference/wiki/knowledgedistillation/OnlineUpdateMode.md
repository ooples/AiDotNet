---
title: "OnlineUpdateMode"
description: "Defines how an online teacher model is updated during training."
section: "API Reference"
---

`Enums` · `AiDotNet.KnowledgeDistillation.Teachers`

Defines how an online teacher model is updated during training.

## Fields

| Field | Summary |
|:-----|:--------|
| `EMA` | Exponential Moving Average - smooth, stable updates. |
| `GradientBased` | Gradient-based updates - standard gradient descent on teacher. |
| `MomentumBased` | Momentum-based updates - teacher follows with momentum. |

