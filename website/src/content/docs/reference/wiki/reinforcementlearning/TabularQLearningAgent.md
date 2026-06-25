---
title: "TabularQLearningAgent<T>"
description: "Tabular Q-Learning agent using lookup table for Q-values."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.Agents.TabularQLearning`

Tabular Q-Learning agent using lookup table for Q-values.

## For Beginners

Q-Learning is like creating a cheat sheet: for every situation (state) and
action you could take, you write down how good that choice is (Q-value).
Over time, you update this sheet based on actual rewards you receive.

Key features:

- **Off-Policy**: Learns optimal policy while following exploratory policy
- **Tabular**: Uses lookup table, no function approximation
- **Model-Free**: Doesn't need to know environment dynamics
- **Value-Based**: Learns action values, derives policy from them

Perfect for: Small discrete state/action spaces (grid worlds, simple games)
Famous for: Watkins 1989, the foundation of modern RL

## How It Works

Tabular Q-Learning is the foundational RL algorithm that maintains a table
of Q-values for each state-action pair. No neural networks required.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TabularQLearningAgent` | Initializes a new instance with default settings. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetOptions` |  |

