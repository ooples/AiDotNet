---
title: "DynaQAgent<T>"
description: "Dyna-Q agent combining learning and planning using a learned model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.Agents.Planning`

Dyna-Q agent combining learning and planning using a learned model.

## For Beginners

Dyna-Q learns from real experiences AND simulated ones.
After each real interaction, it also "replays" past experiences in a mental model,
like practicing chess moves in your head. This lets it learn much faster than
pure Q-learning because each real experience generates many simulated learning updates.
The planning steps parameter controls how many simulated updates happen per real step.

## Methods

| Method | Summary |
|:-----|:--------|
| `GetOptions` |  |

