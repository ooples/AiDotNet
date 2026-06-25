---
title: "AttentionMatchingMode"
description: "Defines how to match attention patterns between teacher and student."
section: "API Reference"
---

`Enums` · `AiDotNet.KnowledgeDistillation.Strategies`

Defines how to match attention patterns between teacher and student.

## Fields

| Field | Summary |
|:-----|:--------|
| `Cosine` | Cosine similarity - focuses on direction/pattern rather than magnitude. |
| `KL` | KL Divergence - treats attention as probability distribution, preserves structure. |
| `MSE` | Mean Squared Error - simple, fast, treats all attention weights equally. |

