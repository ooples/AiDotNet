---
title: "Diagnostics"
description: "All 18 public types in the AiDotNet.diagnostics namespace, organized by kind."
section: "API Reference"
---

**18** public types in this namespace, organized by kind.

## Models & Types (12)

| Type | Summary |
|:-----|:--------|
| [`AccelerationSnapshot`](/docs/reference/wiki/diagnostics/accelerationsnapshot/) | Immutable snapshot of acceleration state at a point in time. |
| [`MemoryDiff`](/docs/reference/wiki/diagnostics/memorydiff/) | Difference between two memory snapshots. |
| [`MemorySnapshot`](/docs/reference/wiki/diagnostics/memorysnapshot/) | A snapshot of memory usage at a point in time. |
| [`OperationTiming`](/docs/reference/wiki/diagnostics/operationtiming/) |  |
| [`ProfileComparison`](/docs/reference/wiki/diagnostics/profilecomparison/) | Result of comparing two profiling reports. |
| [`ProfileRegression`](/docs/reference/wiki/diagnostics/profileregression/) | Details about a regression or improvement in performance. |
| [`ProfileReport`](/docs/reference/wiki/diagnostics/profilereport/) | A comprehensive profiling report containing all collected metrics. |
| [`ProfilerSession`](/docs/reference/wiki/diagnostics/profilersession/) | Instance-based performance profiler for ML operations. |
| [`ProfilerSessionEntry`](/docs/reference/wiki/diagnostics/profilersessionentry/) | A single profiler entry tracking an operation's performance using streaming algorithms. |
| [`ProfilerSessionTimer`](/docs/reference/wiki/diagnostics/profilersessiontimer/) | A manual profiler timer that must be explicitly stopped. |
| [`ProfilerStats`](/docs/reference/wiki/diagnostics/profilerstats/) | Statistics for a profiled operation. |
| [`TensorsOperationProfile`](/docs/reference/wiki/diagnostics/tensorsoperationprofile/) | Structured performance report captured at build time when the builder opts in via `EnableProfiling()`. |

## Interfaces (1)

| Type | Summary |
|:-----|:--------|
| [`IProfilerEntry`](/docs/reference/wiki/diagnostics/iprofilerentry/) | Interface for profiler entries to support both legacy and new implementations. |

## Enums (1)

| Type | Summary |
|:-----|:--------|
| [`MemoryPressureLevel`](/docs/reference/wiki/diagnostics/memorypressurelevel/) | Memory pressure levels. |

## Structs (2)

| Type | Summary |
|:-----|:--------|
| [`MemoryScope`](/docs/reference/wiki/diagnostics/memoryscope/) | A scope that automatically captures before/after memory snapshots. |
| [`ProfilerSessionScope`](/docs/reference/wiki/diagnostics/profilersessionscope/) | A scoped profiler that automatically records timing when disposed. |

## Helpers & Utilities (2)

| Type | Summary |
|:-----|:--------|
| [`AccelerationDiagnostics`](/docs/reference/wiki/diagnostics/accelerationdiagnostics/) | Snapshots the live acceleration environment so users can see which SIMD, GPU, and native BLAS paths are actually engaged at runtime, instead of assuming from config. |
| [`MemoryTracker`](/docs/reference/wiki/diagnostics/memorytracker/) | Tracks memory usage and allocations during ML operations. |

