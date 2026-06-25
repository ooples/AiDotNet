---
title: "GenerationResult<T>"
description: "Result of a generation request."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Serving.ContinuousBatching`

Result of a generation request.

## Properties

| Property | Summary |
|:-----|:--------|
| `FinishReason` | Reason why generation stopped. |
| `GeneratedLength` | Number of tokens generated. |
| `GeneratedTokens` | Only the generated tokens (excluding prompt). |
| `GenerationTime` | Time spent generating. |
| `QueueTime` | Time spent waiting in queue. |
| `SequenceId` | Unique ID of the sequence. |
| `TokenIds` | All token IDs including prompt. |
| `TokensPerSecond` | Generation speed. |

