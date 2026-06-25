---
title: "DistillationLossType"
description: "Types of distillation loss functions."
section: "API Reference"
---

`Enums` · `AiDotNet.ContinualLearning.Strategies`

Types of distillation loss functions.

## Fields

| Field | Summary |
|:-----|:--------|
| `KLDivergence` | KL Divergence - standard distillation loss. |
| `MSE` | Mean Squared Error between soft targets. |
| `SoftCrossEntropy` | Cross-entropy with soft targets. |
| `SymmetricKL` | Symmetric KL Divergence (KL(T\|\|S) + KL(S\|\|T)). |

