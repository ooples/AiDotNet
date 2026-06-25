---
title: "DraftModelType"
description: "Types of draft models for speculative decoding."
section: "API Reference"
---

`Enums` · `AiDotNet.Configuration`

Types of draft models for speculative decoding.

## Fields

| Field | Summary |
|:-----|:--------|
| `Custom` | Custom draft model (internal/serving integration). |
| `NGram` | N-gram based statistical model (fast, no GPU). |
| `SmallNeural` | Small neural network model (more accurate, uses GPU). |

