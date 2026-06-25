---
title: "IPolicy<T>"
description: "Core interface for RL policies - defines how to select actions."
section: "API Reference"
---

`Interfaces` · `AiDotNet.ReinforcementLearning.Policies`

Core interface for RL policies - defines how to select actions.

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeLogProb(Vector<>,Vector<>)` | Computes the log probability of a given action in a given state. |
| `GetNetworks` | Gets the neural networks used by this policy. |
| `Reset` | Resets any internal state (e.g., for recurrent policies, exploration noise). |
| `SelectAction(Vector<>,Boolean)` | Selects an action given the current state. |

