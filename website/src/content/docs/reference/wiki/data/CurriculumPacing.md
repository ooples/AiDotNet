---
title: "CurriculumPacing"
description: "Pacing function controlling how fast the data pool grows."
section: "API Reference"
---

`Enums` · `AiDotNet.Data.Quality`

Pacing function controlling how fast the data pool grows.

## Fields

| Field | Summary |
|:-----|:--------|
| `Exponential` | Exponential growth (slow start, rapid expansion). |
| `Linear` | Linear growth from initial fraction to 1.0. |
| `Step` | Step function: discrete jumps in data pool size. |

