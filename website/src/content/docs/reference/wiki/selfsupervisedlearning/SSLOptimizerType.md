---
title: "SSLOptimizerType"
description: "Optimizer types optimized for SSL training."
section: "API Reference"
---

`Enums` · `AiDotNet.SelfSupervisedLearning`

Optimizer types optimized for SSL training.

## Fields

| Field | Summary |
|:-----|:--------|
| `Adam` | Adam optimizer (good for small batches). |
| `AdamW` | AdamW with decoupled weight decay. |
| `LAMB` | LAMB (Layer-wise Adaptive Moments for Batch training) - good for transformers. |
| `LARS` | LARS (Layer-wise Adaptive Rate Scaling) - recommended for large batch SSL. |
| `SGD` | Standard SGD with momentum. |

