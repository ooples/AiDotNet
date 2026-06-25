---
title: "GpuDiagnosticSink"
description: "Delegate that receives GPU backend diagnostic messages in lieu of `WriteLine`."
section: "API Reference"
---

`Delegates` · `AiDotNet.Configuration`

Delegate that receives GPU backend diagnostic messages in lieu of
`WriteLine`. Applications register a sink
to route GPU diagnostics through their logging framework of choice
(Spectre.Console, Serilog, Microsoft.Extensions.Logging, structured
logs, etc.).

## How It Works

Addresses Option C of github.com/ooples/AiDotNet#1122:
"Use ILogger instead of Console.WriteLine — Then applications control
output through their logging framework."

Forward-compatibility: the sink is captured in
`Sink` immediately. When the underlying
AiDotNet.Tensors package supports sink routing (v0.39+), the diagnostic
messages are delivered to the sink WITH level tagging. On current
Tensors v0.38.0, the sink is stored but the Console.WriteLine calls
in the Tensors layer still go to Console directly; the bool-level
gate (`Verbose`) still suppresses
them when `Silent` or
`Minimal`.

For Microsoft.Extensions.Logging integration, see
`ILogger)`
which wraps an `ILogger` instance as a sink.

