---
title: "TrainingDiagnosticEvent"
description: "Base type for structured training-pipeline diagnostic events."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Configuration`

Base type for structured training-pipeline diagnostic events.
Sinks can dispatch on the runtime type to render or filter
per-event-type without parsing string payloads.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TrainingDiagnosticEvent(TrainingDiagnosticLevel)` | Base type for structured training-pipeline diagnostic events. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ToString` | Default text rendering — sinks that want raw text can call this. |

