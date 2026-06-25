---
title: "SpeculationPolicy"
description: "Policies for enabling/disabling speculative decoding at runtime."
section: "API Reference"
---

`Enums` · `AiDotNet.Configuration`

Policies for enabling/disabling speculative decoding at runtime.

## Fields

| Field | Summary |
|:-----|:--------|
| `Auto` | Automatically decide based on runtime conditions (recommended). |
| `ForceOff` | Always disable speculative decoding even if enabled in config. |
| `ForceOn` | Always enable speculative decoding when configured. |
| `LatencyFirst` | Prefer speculative decoding to reduce latency, even under moderate load. |
| `ThroughputFirst` | Prefer throughput and stability: use speculative decoding only when conditions are ideal. |

