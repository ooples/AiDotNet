---
title: "IExplorationStrategy<T>"
description: "Interface for exploration strategies used by policies."
section: "API Reference"
---

`Interfaces` · `AiDotNet.ReinforcementLearning.Policies.Exploration`

Interface for exploration strategies used by policies.

## Methods

| Method | Summary |
|:-----|:--------|
| `GetExplorationAction(Vector<>,Vector<>,Int32,Random)` | Modifies or replaces the policy's action for exploration. |
| `Reset` | Resets internal state (e.g., for new episodes or training sessions). |
| `Update` | Updates internal parameters (e.g., epsilon decay, noise reduction). |

