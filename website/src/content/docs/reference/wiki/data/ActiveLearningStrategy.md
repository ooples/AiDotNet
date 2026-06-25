---
title: "ActiveLearningStrategy"
description: "Active learning selection strategies."
section: "API Reference"
---

`Enums` · `AiDotNet.Data.Sampling`

Active learning selection strategies.

## Fields

| Field | Summary |
|:-----|:--------|
| `Diversity` | Select diverse samples using distance-based clustering. |
| `Hybrid` | Combine uncertainty and diversity. |
| `Random` | Random sampling (baseline). |
| `Uncertainty` | Select samples with highest uncertainty (e.g., entropy, margin). |

