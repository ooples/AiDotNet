---
title: "PipelineMode"
description: "Specifies how transformers in a pipeline receive their input."
section: "API Reference"
---

`Enums` · `AiDotNet.Preprocessing.TimeSeries`

Specifies how transformers in a pipeline receive their input.

## Fields

| Field | Summary |
|:-----|:--------|
| `Parallel` | Each transformer receives the original input data. |
| `Sequential` | Each transformer receives the previous transformer's output. |

