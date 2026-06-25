---
title: "CurriculumStrategy"
description: "Defines how the curriculum progresses over epochs."
section: "API Reference"
---

`Enums` · `AiDotNet.Data.Sampling`

Defines how the curriculum progresses over epochs.

## Fields

| Field | Summary |
|:-----|:--------|
| `CompetenceBased` | Competence-based: difficulty threshold based on current performance. |
| `Exponential` | Exponential progression (faster ramp-up of difficulty). |
| `Linear` | Linear progression from easy to all samples. |
| `Stepped` | Step-wise progression with discrete difficulty levels. |

