---
title: "GpuDiagnosticsOptions"
description: "Options for controlling GPU backend diagnostic output visibility."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Configuration`

Options for controlling GPU backend diagnostic output visibility.
Addresses github.com/ooples/AiDotNet#1122 — all three requested
controls (environment variable, static configuration, ILogger /
custom sink) are reachable through this options class or the
underlying `GpuDiagnosticsConfig` static facade.

## For Beginners

If your AI application is printing lots of
`[OpenClBackend] Compiling kernels...` messages, pass
`new GpuDiagnosticsOptions { Level = GpuDiagnosticLevel.Silent }`
to the builder's `ConfigureGpuDiagnostics` method. If you want
them routed through your logger instead, set `Sink`.

## How It Works

AiDotNet's GPU backends (OpenCL, HIP, CUDA) emit status messages during
device discovery, kernel compilation, and availability checks. This
options class lets applications configure the verbosity and routing
of that output via the fluent
`GpuDiagnosticsOptions)`
builder method.

All properties are nullable — `null` means "don't change the
current setting", so passing an empty options instance is a no-op.
This matches the AiDotNet facade pattern
(`TelemetryConfig` / `ProfilingConfig`).

## Properties

| Property | Summary |
|:-----|:--------|
| `Level` | Verbosity level. |
| `Sink` | Optional sink that receives diagnostic messages. |
| `Verbose` | Legacy bool-level flag. |

