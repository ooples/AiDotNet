---
title: "SARSAAgent<T>"
description: "SARSA (State-Action-Reward-State-Action) agent using tabular methods."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.Agents.SARSA`

SARSA (State-Action-Reward-State-Action) agent using tabular methods.

## For Beginners

SARSA is like Q-Learning's more cautious cousin. While Q-Learning learns
the optimal policy assuming perfect future actions, SARSA learns based on
what you actually do (including exploratory mistakes).

Key differences from Q-Learning:

- **On-Policy**: Learns from actions it actually takes
- **More Conservative**: Safer in risky environments (cliff walking)
- **Exploration Aware**: Updates reflect exploration strategy
- **Convergence**: Converges to optimal policy only if exploration decreases

Update rule: Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
(Uses actual next action a', not max)

Perfect for: Environments where safety matters, risky state transitions
Famous for: Rummery & Niranjan 1994, on-policy TD control

## How It Works

SARSA is an on-policy TD control algorithm that learns Q-values based on
the action actually taken by the current policy, not the optimal action.

## Methods

| Method | Summary |
|:-----|:--------|
| `GetOptions` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `DefaultStateSize` | Initializes a new instance with default settings. |

