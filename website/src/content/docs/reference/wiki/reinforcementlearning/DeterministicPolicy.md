---
title: "DeterministicPolicy<T>"
description: "Deterministic policy for continuous action spaces."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.Policies`

Deterministic policy for continuous action spaces.
Directly outputs actions without sampling from a distribution.
Commonly used in DDPG, TD3, and other deterministic policy gradient methods.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DeterministicPolicy` | Initializes a new instance with default settings. |
| `DeterministicPolicy(INeuralNetwork<>,Int32,IExplorationStrategy<>,Boolean,Random)` | Initializes a new instance of the DeterministicPolicy class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeLogProb(Vector<>,Vector<>)` | Computes log probability for a deterministic policy. |
| `Dispose(Boolean)` | Disposes of policy resources. |
| `GetNetworks` | Gets the neural networks used by this policy. |
| `Reset` | Resets the exploration strategy. |
| `SelectAction(Vector<>,Boolean)` | Selects a deterministic action from the policy network. |

