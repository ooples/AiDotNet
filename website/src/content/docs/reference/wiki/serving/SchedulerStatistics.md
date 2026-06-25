---
title: "SchedulerStatistics"
description: "Statistics about the scheduler state."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Serving.ContinuousBatching`

Statistics about the scheduler state.

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxCacheSlots` | Maximum number of cache slots. |
| `MaxMemoryBytes` | Maximum memory available (bytes). |
| `MemoryUtilization` | Memory utilization (0-1). |
| `PreemptedSequences` | Number of preempted sequences. |
| `RunningSequences` | Number of sequences currently being processed. |
| `SlotUtilization` | Cache slot utilization (0-1). |
| `TotalSequences` | Total sequences in system. |
| `UsedCacheSlots` | Number of cache slots in use. |
| `UsedMemoryBytes` | Memory currently in use (bytes). |
| `WaitingSequences` | Number of sequences waiting to be processed. |

