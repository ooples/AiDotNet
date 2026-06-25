---
title: "ProfilerStats"
description: "Statistics for a profiled operation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diagnostics`

Statistics for a profiled operation.

## Properties

| Property | Summary |
|:-----|:--------|
| `AllocationCount` | Number of allocation events. |
| `Count` | Number of times the operation was called. |
| `MaxMs` | Maximum time in milliseconds. |
| `MeanMs` | Mean time in milliseconds. |
| `MinMs` | Minimum time in milliseconds. |
| `Name` | Operation name. |
| `OpsPerSecond` | Gets the operations per second based on count and total time. |
| `P50Ms` | 50th percentile (median) in milliseconds. |
| `P95Ms` | 95th percentile in milliseconds. |
| `P99Ms` | 99th percentile in milliseconds. |
| `Parents` | Parent operations (for hierarchy tracking). |
| `StdDevMs` | Standard deviation in milliseconds. |
| `TotalAllocations` | Total bytes allocated. |
| `TotalMs` | Total time in milliseconds. |

