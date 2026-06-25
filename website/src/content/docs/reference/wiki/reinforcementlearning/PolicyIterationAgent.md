---
title: "PolicyIterationAgent<T>"
description: "Policy Iteration agent for reinforcement learning using dynamic programming."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.Agents.DynamicProgramming`

Policy Iteration agent for reinforcement learning using dynamic programming.

## For Beginners

Policy Iteration works in two repeating steps:
1) Evaluate: calculate how good the current strategy is for every state
2) Improve: update the strategy to be greedy with respect to the evaluation
These steps repeat until the strategy stops changing (convergence). It requires a
complete model of the environment (transition probabilities and rewards) and works best
for small, fully-known environments like grid worlds or simple games.

## How It Works

Policy Iteration alternates between policy evaluation and policy improvement
until convergence to the optimal policy.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PolicyIterationAgent` | Initializes a new instance with default settings. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetOptions` |  |

