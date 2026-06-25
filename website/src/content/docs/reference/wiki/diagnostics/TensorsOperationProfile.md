---
title: "TensorsOperationProfile"
description: "Structured performance report captured at build time when the builder opts in via `EnableProfiling()`."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diagnostics`

Structured performance report captured at build time when the builder opts in via
`EnableProfiling()`. Wraps Tensors' `PerformanceProfiler` output
so callers don't have to dip into Tensors internals for a timing breakdown.

## Properties

| Property | Summary |
|:-----|:--------|
| `Operations` | Per-operation timing statistics, sorted by total time descending. |
| `RawReport` | The profiler's raw text report (via `PerformanceProfiler.GenerateReport`). |
| `TotalMilliseconds` | Total wall-clock time across every profiled operation (ms). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Capture` | Builds a ProfilingReport from the live Tensors `Instance`. |

