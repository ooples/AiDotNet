---
title: "QMIXAgent<T>"
description: "QMIX agent for multi-agent value-based reinforcement learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.Agents.QMIX`

QMIX agent for multi-agent value-based reinforcement learning.

## For Beginners

QMIX solves multi-agent problems by letting each agent learn its own Q-values,
then using a "mixing network" to combine them into a team Q-value.

Key innovation:

- **Value Factorization**: Team value = mix(agent1_Q, agent2_Q, ...)
- **Mixing Network**: Ensures individual and joint actions are consistent
- **Monotonicity**: If one agent improves, team improves
- **Decentralized Execution**: Each agent acts independently

Think of it like: Each player estimates their contribution, and a coach
combines these to determine the team's overall score.

Famous for: StarCraft II micromanagement, cooperative games

## How It Works

QMIX factorizes joint action-values into per-agent values using a mixing network
that monotonically combines them.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `QMIXAgent` | Initializes a new instance with default settings. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetOptions` |  |
| `SelectActionForAgent(Int32,Vector<>,Boolean)` | Select action for a specific agent using epsilon-greedy. |
| `StoreMultiAgentExperience(List<Vector<>>,List<Vector<>>,,List<Vector<>>,Vector<>,Vector<>,Boolean)` | Store multi-agent experience with global state. |

