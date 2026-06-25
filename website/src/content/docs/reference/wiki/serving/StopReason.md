---
title: "StopReason"
description: "Reasons why generation stopped."
section: "API Reference"
---

`Enums` · `AiDotNet.Serving.ContinuousBatching`

Reasons why generation stopped.

## Fields

| Field | Summary |
|:-----|:--------|
| `Cancelled` | Request was cancelled by user. |
| `EndOfSequence` | Generated end-of-sequence token. |
| `Error` | An error occurred during generation. |
| `MaxLength` | Reached maximum token limit. |
| `StopToken` | Generated a stop token. |

