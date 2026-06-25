---
title: "UncertaintyPolicy"
description: "Policy for handling uncertain labels in CheXpert."
section: "API Reference"
---

`Enums` · `AiDotNet.Data.Vision.Benchmarks`

Policy for handling uncertain labels in CheXpert.

## Fields

| Field | Summary |
|:-----|:--------|
| `Ignore` | Exclude samples with uncertain labels from training entirely. |
| `Ones` | Treat uncertain labels as positive (1). |
| `Zeros` | Treat uncertain labels as negative (0). |

