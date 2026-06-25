---
title: "BufferStatistics"
description: "Statistics about the experience replay buffer."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ContinualLearning.Memory`

Statistics about the experience replay buffer.

## Properties

| Property | Summary |
|:-----|:--------|
| `AveragePriority` | Average priority of stored samples. |
| `Count` | Current number of stored examples. |
| `EstimatedMemoryBytes` | Estimated memory usage in bytes. |
| `FillRatio` | Ratio of current count to max size. |
| `MaxSize` | Maximum buffer capacity. |
| `TaskCount` | Number of distinct tasks in the buffer. |
| `TaskDistribution` | Number of examples per task. |
| `TotalReplaySamples` | Total samples returned via replay. |
| `TotalSamplesProcessed` | Total samples ever added to the buffer. |

