---
title: "ScoredMemory"
description: "A memory paired with its relevance score for a particular query, as returned by `CancellationToken)`."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Memory`

A memory paired with its relevance score for a particular query, as returned by
`CancellationToken)`.

## For Beginners

When the assistant searches its notes, each matching note comes back with a
number saying how well it matched (higher = more relevant). This pairs the note with that number.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ScoredMemory(AgentMemory,Double)` | Initializes a new scored memory. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Memory` | Gets the matched memory. |
| `Score` | Gets the relevance score (higher is more relevant). |

