---
title: "MemoryTracker"
description: "Tracks memory usage and allocations during ML operations."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Diagnostics`

Tracks memory usage and allocations during ML operations.

## How It Works

**Features:**

- GC heap tracking
- Working set monitoring
- Allocation rate calculation
- Memory snapshot comparison

**Usage:**

## Properties

| Property | Summary |
|:-----|:--------|
| `IsEnabled` | Gets whether memory tracking is enabled. |
| `MaxHistorySize` | Gets or sets the maximum history size (default: 10000). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Disable` | Disables memory tracking. |
| `Enable` | Enables memory tracking. |
| `EstimateKVCacheMemory(Int32,Int32,Int32,Int32,Int32,Int32)` | Estimates KV-cache memory for a model configuration. |
| `EstimateTensorMemory(Int32[],Int32)` | Estimates the memory footprint of a tensor. |
| `GetHistory` | Gets all recorded snapshots. |
| `GetPressureLevel` | Gets the current memory pressure level. |
| `Reset` | Clears all recorded history. |
| `Snapshot(String,Boolean)` | Takes a snapshot of current memory usage. |
| `TrackScope(String)` | Creates a memory tracking scope that records before/after snapshots. |

