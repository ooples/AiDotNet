---
title: "TrainingDiagnosticSink"
description: "Delegate that receives training-pipeline diagnostic events in lieu of the default `String)` path."
section: "API Reference"
---

`Delegates` · `AiDotNet.Configuration`

Delegate that receives training-pipeline diagnostic events in lieu of
the default `String)` path.
Applications register a sink to route training diagnostics through
their logging framework of choice (Spectre.Console, Serilog,
Microsoft.Extensions.Logging, structured logs, OpenTelemetry, etc.).

## How It Works

Sink exceptions are caught by
`TrainingDiagnosticEvent)` and reported via
`String)` so a
throwing sink cannot break training. There is no opt-in to
rethrow / fail-fast semantics today — if a caller needs that, file
a feature request. Sinks should still avoid throwing in hot-path
instrumentation: the Trace.TraceError fallback is synchronous and
runs in the training step's critical path.

