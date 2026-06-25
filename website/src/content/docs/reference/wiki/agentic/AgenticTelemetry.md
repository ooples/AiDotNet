---
title: "AgenticTelemetry"
description: "The instrumentation source for the agentic subsystem: a named `ActivitySource` (traces) and `Meter` (metrics) following OpenTelemetry GenAI semantic conventions."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Agentic.Pipeline`

The instrumentation source for the agentic subsystem: a named `ActivitySource` (traces) and
`Meter` (metrics) following OpenTelemetry GenAI semantic conventions. Any OpenTelemetry
exporter subscribed to the source name `SourceName` collects chat spans and token metrics with
no extra wiring.

## For Beginners

This is the labeled channel the system broadcasts "what the model did" on —
timings, token counts, finish reasons. Point your monitoring/dashboard at the name and you get visibility
without changing agent code.

## How It Works

A named `ActivitySource`/`Meter` is the idiomatic .NET instrumentation pattern:
emission is essentially free when nothing is listening, and standard OpenTelemetry collectors enable it by
name. `TelemetryChatMiddleware` is the producer.

## Fields

| Field | Summary |
|:-----|:--------|
| `Meter` | The meter that emits chat metrics. |
| `OperationCount` | Counts chat operations (GenAI convention: gen_ai.client.operation.count). |
| `Source` | The activity source that emits chat spans. |
| `SourceName` | The OpenTelemetry source/meter name for the agentic subsystem. |
| `TokenUsage` | Records token usage per call (GenAI convention: gen_ai.client.token.usage). |

