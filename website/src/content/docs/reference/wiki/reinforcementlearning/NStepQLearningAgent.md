---
title: "NStepQLearningAgent<T>"
description: "N-step Q-Learning agent using multi-step off-policy returns."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.Agents.NStepQLearning`

N-step Q-Learning agent using multi-step off-policy returns.

## For Beginners

N-step Q-Learning extends regular Q-Learning by looking ahead
n steps before bootstrapping. Regular Q-Learning updates based on the very next reward,
while n-step looks at n future rewards. This propagates information faster (like seeing
further down a road before deciding which way to turn). The trade-off: higher n means
faster learning but more variance. Common values are n=3 to n=10.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NStepQLearningAgent` | Initializes a new instance with default settings. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetOptions` |  |

