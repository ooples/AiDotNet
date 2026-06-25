---
title: "PrioritizedSweepingAgent<T>"
description: "Prioritized Sweeping agent that focuses planning on high-priority state-actions."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.Agents.Planning`

Prioritized Sweeping agent that focuses planning on high-priority state-actions.

## For Beginners

Prioritized Sweeping is like Dyna-Q but smarter about which
simulated updates to do. Instead of replaying random past experiences, it prioritizes
updates where the biggest changes happened. Think of it like studying: focus on the
topics where you got the most wrong. A priority queue tracks which state-actions need
the most urgent updates, making planning much more efficient.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PrioritizedSweepingAgent` | Initializes a new instance with default settings. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetOptions` |  |

