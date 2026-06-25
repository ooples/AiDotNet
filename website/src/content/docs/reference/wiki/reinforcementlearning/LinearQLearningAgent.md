---
title: "LinearQLearningAgent<T>"
description: "Linear Q-Learning agent using linear function approximation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.Agents.AdvancedRL`

Linear Q-Learning agent using linear function approximation.

## For Beginners

Linear Q-Learning replaces the Q-table with a linear function
Q(s,a) = w dot phi(s,a), where phi extracts features from state-action pairs. This allows
handling continuous or large state spaces that would be impossible with tables. Think of it
like using a formula instead of a lookup table. The trade-off is that it can only represent
linear relationships, but it scales to much larger problems than tabular Q-learning.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LinearQLearningAgent` | Initializes a new instance with default settings. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetOptions` |  |

