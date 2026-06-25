---
title: "ProfilerSession"
description: "Instance-based performance profiler for ML operations."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diagnostics`

Instance-based performance profiler for ML operations.

## How It Works

**Overview:**
ProfilerSession provides comprehensive performance monitoring for machine learning
workloads including timing, memory tracking, and hierarchical call analysis.

**Facade Pattern:**
This class follows the AiDotNet facade pattern. Users don't create ProfilerSession directly;
instead, they configure profiling through `AiModelBuilder.ConfigureProfiling()`
and access results through `AiModelResult.ProfilingReport`.

**Production-Ready Features:**

- O(1) streaming statistics using Welford's algorithm
- Bounded memory with reservoir sampling for percentiles
- Configurable sampling rate for high-frequency operations
- Thread-safe timing collection
- Hierarchical call tree tracking
- Memory allocation monitoring
- Statistical aggregation (min, max, mean, p95, p99)
- Profile scope pattern (using blocks)

**Internal Usage Example:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ProfilerSession(ProfilingConfig)` | Creates a new profiler session with the specified configuration. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Config` | Gets the configuration for this profiler session. |
| `Elapsed` | Gets the elapsed time since this session started. |
| `IsEnabled` | Gets whether the profiler session is currently enabled. |
| `OperationCount` | Gets the total number of unique operations tracked. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Disable` | Disables profiling for this session. |
| `Enable` | Enables profiling for this session. |
| `GetCallStack` | Gets the current call stack for hierarchical tracking. |
| `GetOperationNames` | Gets all operation names that have been profiled. |
| `GetReport` | Gets a comprehensive profiling report. |
| `GetStats(String)` | Gets timing statistics for a specific operation. |
| `GetSummary` | Gets a summary string of profiling results. |
| `RecordAllocation(String,Int64)` | Records a memory allocation. |
| `RecordTiming(String,TimeSpan,String)` | Records a timing sample for a named operation. |
| `Reset` | Resets all collected profiling data. |
| `Scope(String)` | Creates a scoped profiler that automatically records duration. |
| `Start(String)` | Starts a manual profiler timer. |

