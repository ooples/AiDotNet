---
title: "SequenceStatus"
description: "Status of a sequence in the continuous batching system."
section: "API Reference"
---

`Enums` · `AiDotNet.Serving.ContinuousBatching`

Status of a sequence in the continuous batching system.

## Fields

| Field | Summary |
|:-----|:--------|
| `Cancelled` | Sequence was cancelled. |
| `Completed` | Sequence has completed generation. |
| `Failed` | Sequence encountered an error. |
| `Generating` | Sequence is actively generating tokens. |
| `Paused` | Sequence is paused (preempted for higher priority). |
| `Pending` | Sequence is waiting to be processed. |
| `Prefilling` | Sequence is being prefilled (processing prompt). |

