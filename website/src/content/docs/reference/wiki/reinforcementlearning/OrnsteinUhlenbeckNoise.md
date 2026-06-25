---
title: "OrnsteinUhlenbeckNoise<T>"
description: "Ornstein-Uhlenbeck process noise for temporally correlated exploration."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.Policies.Exploration`

Ornstein-Uhlenbeck process noise for temporally correlated exploration.
Commonly used in DDPG and other continuous control algorithms.
Process equation: dx = θ(μ - x)dt + σdW

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OrnsteinUhlenbeckNoise` | Initializes a new instance of the Ornstein-Uhlenbeck noise exploration. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetExplorationAction(Vector<>,Vector<>,Int32,Random)` | Applies Ornstein-Uhlenbeck noise to the policy action. |
| `Reset` | Resets the noise state to zero. |
| `Update` | Updates internal parameters (no-op for OU noise as it self-regulates). |

