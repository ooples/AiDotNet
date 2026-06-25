---
title: "BatcherStatistics"
description: "Statistics about the batcher's operation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Serving.ContinuousBatching`

Statistics about the batcher's operation.

## Properties

| Property | Summary |
|:-----|:--------|
| `AverageBatchSize` | Average batch size per iteration. |
| `MemoryUtilization` | Memory utilization (0-1). |
| `RequestsPerSecond` | Requests completed per second. |
| `RunningRequests` | Requests currently being processed. |
| `RuntimeSeconds` | Total runtime in seconds. |
| `TokensPerSecond` | Tokens generated per second. |
| `TotalIterations` | Total batching iterations. |
| `TotalRequestsProcessed` | Total requests completed since start. |
| `TotalTokensGenerated` | Total tokens generated since start. |
| `WaitingRequests` | Requests currently waiting. |

