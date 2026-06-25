---
title: "GpuDiagnosticsLoggerExtensions"
description: "Extension methods that adapt `ILogger` into a `GpuDiagnosticSink`."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Configuration`

Extension methods that adapt
`ILogger` into a
`GpuDiagnosticSink`. Addresses Option C of
github.com/ooples/AiDotNet#1122:
"Use ILogger instead of Console.WriteLine — Then applications control
output through their logging framework."

## Methods

| Method | Summary |
|:-----|:--------|
| `ToSink(ILogger)` | Wraps an `ILogger` as a `GpuDiagnosticSink`, mapping `GpuDiagnosticLevel` onto `LogLevel`: `Silent` → no emission (sink never fires when level is Silent because the level-gate upstream drops the message), `Minimal` → `Information`, `Verbose… |

