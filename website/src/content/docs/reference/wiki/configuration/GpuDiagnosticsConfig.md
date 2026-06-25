---
title: "GpuDiagnosticsConfig"
description: "Process-global control for GPU backend diagnostic output visibility."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Configuration`

Process-global control for GPU backend diagnostic output visibility.
Exposes three orthogonal knobs matching
github.com/ooples/AiDotNet#1122's requested-change checklist:
environment variable, static configuration, and ILogger / custom sink.

## How It Works

AiDotNet's GPU backends (OpenCL, HIP, CUDA) emit status messages during
device discovery, kernel compilation, and availability checks.
Historically these were always written to `WriteLine`,
producing ~40 lines of output on every `AiModelBuilder.BuildAsync()`.
This class provides the AiDotNet-side facade for controlling that output.

**Three orthogonal controls (all work simultaneously):**

## Properties

| Property | Summary |
|:-----|:--------|
| `Level` | Current verbosity level. |
| `Sink` | Optional sink that receives diagnostic messages when set. |
| `Verbose` | Legacy bool flag. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Emit(GpuDiagnosticLevel,String)` | Emits a diagnostic message, respecting the current `Level` and routing through `Sink` if set (else Console). |
| `PopLevel` | Pops the most recently pushed level from the stack and restores it as the active level. |
| `PushLevel(GpuDiagnosticLevel)` | Push a scoped override of `Level` that automatically restores the previous value when the returned `IDisposable` is disposed (typically via `using var _ = PushLevel(...)`). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_levelStack` | LIFO stack of previously-active levels. |
| `_pushLockSync` | Push-lock for the level stack. |

