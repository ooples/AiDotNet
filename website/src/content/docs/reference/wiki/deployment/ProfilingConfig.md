---
title: "ProfilingConfig"
description: "Configuration for performance profiling during model training and inference."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Deployment.Configuration`

Configuration for performance profiling during model training and inference.

## For Beginners

Profiling measures how long different parts of your ML code take to run.
Think of it like a stopwatch for your code - it helps you find bottlenecks and optimize performance.

What gets tracked:

- Operation timing: How long each training step, forward pass, backward pass takes
- Memory allocations: How much memory is used during training
- Call hierarchy: Which operations call which other operations
- Percentiles: P50 (median), P95, P99 timing for statistical analysis

Why it's important:

- Find bottlenecks in your training pipeline
- Compare performance before and after optimizations
- Identify memory-intensive operations
- Understand your model's computational profile

The profiler uses production-ready algorithms:

- Welford's algorithm for O(1) streaming mean/variance
- Reservoir sampling for bounded-memory percentile estimation
- Configurable sampling for high-frequency operations

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoEnableInDebug` | Gets or sets whether to auto-enable profiling in debug builds (default: true). |
| `CustomTags` | Gets or sets custom tags to include with profiling data. |
| `DetailedTiming` | Gets or sets whether to include detailed timing breakdowns (default: false). |
| `Enabled` | Gets or sets whether profiling is enabled (default: false). |
| `MaxOperations` | Gets or sets the maximum number of unique operations to track (default: 10000). |
| `MeasureMemory` | Alias for TrackAllocations for more intuitive access. |
| `ReservoirSize` | Gets or sets the reservoir size for percentile estimation (default: 1000). |
| `SamplingRate` | Gets or sets the sampling rate for high-frequency operations (default: 1.0 = 100%). |
| `TraceExecution` | Alias for TrackCallHierarchy for more intuitive access. |
| `TrackAllocations` | Gets or sets whether to track memory allocations (default: true). |
| `TrackCallHierarchy` | Gets or sets whether to track parent-child call relationships (default: true). |

