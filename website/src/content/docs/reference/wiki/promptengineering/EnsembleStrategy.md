---
title: "EnsembleStrategy<T>"
description: "Strategy for combining ensemble results."
section: "API Reference"
---

`Enums` · `AiDotNet.PromptEngineering.Optimization`

Strategy for combining ensemble results.

## Fields

| Field | Summary |
|:-----|:--------|
| `BestWins` | Select the single best result across all optimizers. |
| `Parallel` | Run in parallel and pick best result. |
| `Sequential` | Run sequentially, each starting from previous best. |

