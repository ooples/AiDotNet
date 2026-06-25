---
title: "MixedPolicy<T>"
description: "Policy for environments with both discrete and continuous action spaces."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.Policies`

Policy for environments with both discrete and continuous action spaces.
Outputs both categorical distribution for discrete actions and Gaussian for continuous actions.
Common in robotics where you have discrete mode selection and continuous parameter control.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MixedPolicy` | Initializes a new instance of the MixedPolicy class. |
| `MixedPolicy(INeuralNetwork<>,INeuralNetwork<>,Int32,Int32,IExplorationStrategy<>,IExplorationStrategy<>,Boolean,Random)` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeLogProb(Vector<>,Vector<>)` | Computes log probability of mixed action. |
| `GetNetworks` | Gets the neural networks used by this policy. |
| `Reset` | Resets both exploration strategies. |
| `SelectAction(Vector<>,Boolean)` | Selects mixed action: [discrete_action, continuous_actions] |

