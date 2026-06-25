---
title: "AccelerationDiagnostics"
description: "Snapshots the live acceleration environment so users can see which SIMD, GPU, and native BLAS paths are actually engaged at runtime, instead of assuming from config."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Diagnostics`

Snapshots the live acceleration environment so users can see which SIMD, GPU, and
native BLAS paths are actually engaged at runtime, instead of assuming from config.

## How It Works

Wraps Tensors' `PlatformDetector` and `NativeLibraryDetector`
into a single facade-friendly report that can be logged at builder time and surfaced
on `PredictionModelResult` for production observability.

## Methods

| Method | Summary |
|:-----|:--------|
| `GetSnapshot` | Gets a structured snapshot of the current acceleration environment. |

