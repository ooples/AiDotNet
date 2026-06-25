---
title: "BetaPolicy<T>"
description: "Policy using Beta distribution for bounded continuous action spaces."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.Policies`

Policy using Beta distribution for bounded continuous action spaces.
Network outputs alpha and beta parameters for each action dimension.
Actions are naturally bounded to [0, 1] and can be scaled to any [min, max] range.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BetaPolicy` | Initializes a new instance of the BetaPolicy class. |
| `BetaPolicy(INeuralNetwork<>,Int32,IExplorationStrategy<>,Double,Double,Random)` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeLogProb(Vector<>,Vector<>)` | Computes the log probability of an action under the Beta distribution policy. |
| `GetNetworks` | Gets the neural networks used by this policy. |
| `Reset` | Resets the exploration strategy. |
| `SelectAction(Vector<>,Boolean)` | Selects an action by sampling from Beta distributions. |

